from typing import Dict, Tuple

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
        item_index_map: Dict[int, str],
    ):
        super(TopKOfflineREINFORCE, self).__init__()

        self.state_network = state_network
        self.action_policy = action_policy
        self.behavior_policy = behavior_policy

        self.num_actions = self.action_policy.action_space.size(0)

        self.K = K

        self.weight_cap = weight_cap

        self.kl_div_loss = KLDivLoss(reduction="batchmean")

        self.action_policy_optimizer = action_policy_optimizer
        self.behavior_policy_optimizer = behavior_policy_optimizer

        self.item_index_map = item_index_map

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
        action: torch.LongTensor,
        episodic_return: torch.FloatTensor,
    ) -> torch.FloatTensor:
        batch_size = episodic_return.size(0)
        if isinstance(self.action_policy.softmax, nn.AdaptiveLogSoftmaxWithLoss):
            log_action_policy_probs = self.action_policy(state)
            log_action_policy_prob = torch.cat(
                [
                    log_action_policy_probs[batch_idx][action[batch_idx]]
                    for batch_idx in range(batch_size)
                ]
            ).view(batch_size, -1)
            action_policy_prob = torch.exp(log_action_policy_prob)
        else:
            action_policy_probs = self.action_policy(state)
            action_policy_prob = torch.cat(
                [
                    action_policy_probs[batch_idx][action[batch_idx]]
                    for batch_idx in range(batch_size)
                ]
            ).view(batch_size, -1)
            log_action_policy_prob = torch.log(action_policy_prob)

        if isinstance(self.behavior_policy.softmax, nn.AdaptiveLogSoftmaxWithLoss):
            log_behavior_policy_probs = self.behavior_policy(state)
            log_behavior_policy_prob = torch.cat(
                [
                    log_behavior_policy_probs[batch_idx][action[batch_idx]]
                    for batch_idx in range(batch_size)
                ]
            ).view(batch_size, -1)
            behavior_policy_prob = torch.exp(log_behavior_policy_prob)
        else:
            behavior_policy_probs = self.behavior_policy(state)
            behavior_policy_prob = torch.cat(
                [
                    behavior_policy_probs[batch_idx][action[batch_idx]]
                    for batch_idx in range(batch_size)
                ]
            ).view(batch_size, -1)

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
        self, state: torch.FloatTensor, action: torch.LongTensor
    ) -> torch.FloatTensor:
        batch_size = action.size(0)
        if isinstance(self.behavior_policy.softmax, nn.AdaptiveLogSoftmaxWithLoss):
            log_behavior_policy_probs = self.behavior_policy(state)
        else:
            behavior_policy_probs = self.behavior_policy(state)
            log_behavior_policy_probs = torch.log(behavior_policy_probs)

        batch_size = action.size(0)
        actual_action_probs = torch.zeros_like(log_behavior_policy_probs)
        for batch_idx in range(batch_size):
            actual_action_probs[batch_idx][action[batch_idx]] = 1.0

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
    ) -> Tuple[torch.IntTensor, torch.FloatTensor]:
        if isinstance(self.action_policy.softmax, nn.AdaptiveLogSoftmaxWithLoss):
            log_action_policy_probs = self.action_policy(state)
            action_policy_probs = torch.exp(log_action_policy_probs)
        else:
            action_policy_probs = self.action_policy(state)

        sorted_indices = action_policy_probs.argsort(dim=1, descending=True)
        items = sorted_indices[:, : self.K]
        logits = torch.gather(input=action_policy_probs, dim=1, index=items)

        return items, logits
