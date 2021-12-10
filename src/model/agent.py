from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn.modules.loss import KLDivLoss
from torch.optim.optimizer import Optimizer

from .nn import StateTransitionNetwork
from .policy import SoftmaxStochasticPolicy


class TopKOfflineREINFORCE(nn.Module):
    def __init__(
        self,
        state_network: StateTransitionNetwork,
        action_policy: SoftmaxStochasticPolicy,
        behavior_policy: SoftmaxStochasticPolicy,
        action_policy_optimizer: Optimizer,
        behavior_policy_optimizer: Optimizer,
        K: int,
        weight_cap: float,
    ):
        super(TopKOfflineREINFORCE, self).__init__()

        self.state_network = state_network
        self.action_policy = action_policy
        self.behavior_policy = behavior_policy

        self.K = K

        self.weight_cap = weight_cap

        if self.behavior_policy.adaptive_softmax is False:
            self.kl_div_loss = KLDivLoss(reduction="batchmean")

        self.action_policy_optimizer = action_policy_optimizer
        self.behavior_policy_optimizer = behavior_policy_optimizer

    def _compute_lambda_K(self, policy_prob: torch.FloatTensor) -> torch.FloatTensor:
        return self.K * ((1 - policy_prob) ** (self.K - 1))

    def _compute_importance_weight(
        self,
        action_policy_prob: torch.FloatTensor,
        behavior_policy_prob: torch.FloatTensor,
    ) -> torch.FloatTensor:
        weight = torch.div(action_policy_prob, behavior_policy_prob)
        return torch.minimum(
            weight, torch.mul(torch.ones_like(weight), self.weight_cap)
        )

    def action_policy_loss(
        self,
        state: torch.FloatTensor,
        item_index: torch.LongTensor,
        episodic_return: torch.FloatTensor,
    ) -> torch.FloatTensor:
        log_action_policy_prob = self.action_policy(state, item_index)
        action_policy_prob = torch.exp(log_action_policy_prob)

        behavior_policy_prob = torch.exp(
            self.behavior_policy(state.detach(), item_index)
        )

        _lambda_K = self._compute_lambda_K(policy_prob=action_policy_prob)
        _importance_weight = self._compute_importance_weight(
            action_policy_prob=action_policy_prob,
            behavior_policy_prob=behavior_policy_prob,
        )

        return -torch.mean(
            _importance_weight * _lambda_K * episodic_return * log_action_policy_prob,
            dim=0,
        )

    def behavior_policy_loss(
        self, state: torch.FloatTensor, item_index: torch.LongTensor
    ) -> torch.FloatTensor:
        batch_size = item_index.size(0)
        if self.behavior_policy.adaptive_softmax is True:
            out = self.behavior_policy.softmax(state.detach(), item_index.squeeze())
            return out.loss
        else:
            assert state.size(-1) == self.behavior_policy.item_embeddings.weight.size(
                -1
            ), "State & item embedding vector size should match."
            items_embedded = self.behavior_policy.item_embeddings(
                self.behavior_policy.item_space
            )
            logits = torch.stack(
                [
                    torch.sum(s * items_embedded / self.behavior_policy.T, dim=1)
                    for s in state.detach()
                ]
            )
            log_behavior_policy_probs = torch.log(self.behavior_policy.softmax(logits))
            actual_action_probs = log_behavior_policy_probs.new_zeros(
                log_behavior_policy_probs.size()
            )
            for batch_idx in range(batch_size):
                actual_action_probs[batch_idx][item_index[batch_idx]] = 1.0
            return self.kl_div_loss(log_behavior_policy_probs, actual_action_probs)

    def update(
        self,
        action_policy_loss: torch.FloatTensor,
        behavior_policy_loss: torch.FloatTensor,
    ):
        self.action_policy_optimizer.zero_grad()
        self.behavior_policy_optimizer.zero_grad()

        action_policy_loss.backward(retain_graph=True)
        behavior_policy_loss.backward()

        self.action_policy_optimizer.step()
        self.behavior_policy_optimizer.step()

    def recommend(
        self, state: torch.FloatTensor
    ) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor]]:
        if self.action_policy.adaptive_softmax is True:
            log_action_policy_probs = self.action_policy.log_probs(state)
            action_policy_probs = torch.exp(log_action_policy_probs)
        else:
            assert state.size(-1) == self.action_policy.item_embeddings.weight.size(
                -1
            ), "State & item embedding vector size should match."
            items_embedded = self.action_policy.item_embeddings(
                self.action_policy.item_space
            )
            logits = torch.stack(
                [
                    torch.sum(s * items_embedded / self.action_policy.T, dim=1)
                    for s in state.detach()
                ]
            )
            action_policy_probs = self.action_policy.softmax(logits)

        sorted_indices = action_policy_probs.argsort(dim=1, descending=True)
        indexed_items = sorted_indices[:, : self.K]
        logits = torch.gather(input=action_policy_probs, dim=1, index=indexed_items)

        return list(indexed_items), list(logits)

    def to(self, device: torch.device):
        if self.action_policy.adaptive_softmax is True:
            return super().to(device)
        else:
            on_device = super().to(device)
            on_device.action_policy.item_space = on_device.action_policy.item_space.to(
                device
            )
            on_device.behavior_policy.item_space = (
                on_device.behavior_policy.item_space.to(device)
            )
            return on_device
