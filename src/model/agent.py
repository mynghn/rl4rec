from typing import Tuple

import torch
import torch.nn as nn
from torch.nn.modules.loss import KLDivLoss
from torch.optim.optimizer import Optimizer

from .nn import StateTransitionNetwork
from .policy import SoftmaxStochasticPolicy


class TopKOfflineREINFORCE(nn.Module):
    def __init__(
        self,
        pi_state_network: StateTransitionNetwork,
        action_policy: SoftmaxStochasticPolicy,
        beta_state_network: StateTransitionNetwork,
        behavior_policy: SoftmaxStochasticPolicy,
        action_policy_optimizer: Optimizer,
        behavior_policy_optimizer: Optimizer,
        weight_cap: float,
        K: int,
    ):
        super(TopKOfflineREINFORCE, self).__init__()

        self.pi_state_network = pi_state_network
        self.action_policy = action_policy
        self.beta_state_network = beta_state_network
        self.behavior_policy = behavior_policy

        self.weight_cap = weight_cap
        self.K = K

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
        pi_state: torch.FloatTensor,
        beta_state: torch.FloatTensor,
        item_index: torch.LongTensor,
        episodic_return: torch.FloatTensor,
    ) -> torch.FloatTensor:
        log_action_policy_prob = self.action_policy(pi_state, item_index)
        action_policy_prob = torch.exp(log_action_policy_prob)

        behavior_policy_prob = torch.exp(
            self.behavior_policy(beta_state.detach(), item_index)
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
        self, beta_state: torch.FloatTensor, item_index: torch.LongTensor
    ) -> torch.FloatTensor:
        batch_size = item_index.size(0)
        if self.behavior_policy.adaptive_softmax is True:
            out = self.behavior_policy.softmax(beta_state, item_index.squeeze())
            return out.loss
        else:
            assert beta_state.size(
                -1
            ) == self.behavior_policy.item_embeddings.weight.size(
                -1
            ), "State & item embedding vector size should match."
            items_embedded = self.behavior_policy.item_embeddings(
                self.behavior_policy.item_space
            )
            logits = torch.stack(
                [
                    torch.sum(s * items_embedded / self.behavior_policy.T, dim=1)
                    for s in beta_state
                ]
            )
            log_behavior_policy_probs = torch.log(self.behavior_policy.softmax(logits))
            actual_action_probs = log_behavior_policy_probs.new_zeros(
                log_behavior_policy_probs.size()
            )
            for batch_idx in range(batch_size):
                actual_action_probs[batch_idx][item_index[batch_idx]] = 1.0
            return self.kl_div_loss(log_behavior_policy_probs, actual_action_probs)

    def update_action_policy(self, action_policy_loss: torch.FloatTensor):
        self.action_policy_optimizer.zero_grad()
        action_policy_loss.backward()
        self.action_policy_optimizer.step()

    def update_behavior_policy(self, behavior_policy_loss: torch.FloatTensor):
        self.behavior_policy_optimizer.zero_grad()
        behavior_policy_loss.backward()
        self.behavior_policy_optimizer.step()

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

    def get_top_recommendations(
        self, state: torch.FloatTensor, M: int
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        if self.action_policy.adaptive_softmax is True:
            action_policy_probs = torch.exp(self.action_policy.log_probs(state))
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
        items = sorted_indices[:, :M]
        probs = torch.gather(input=action_policy_probs, dim=1, index=items)

        return items, probs

    def recommend(
        self, state: torch.FloatTensor, M: int, K_prime: int
    ) -> torch.LongTensor:
        ordered_item_pool, probs = self.get_top_recommendations(state=state, M=M)
        if K_prime > 0:
            fixed = ordered_item_pool[:, :K_prime]
            sub_item_pool = ordered_item_pool[:, K_prime:]
            sub_probs = probs[:, K_prime:]
            sampled = sub_item_pool[
                sub_probs.multinomial(num_samples=self.K - K_prime, replacement=False)
            ]
            recommendations = torch.cat((fixed, sampled), dim=1)
        else:
            recommendations = ordered_item_pool[
                probs.multinomial(num_samples=self.K, replacement=False)
            ]
        return recommendations

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

    def get_corrected_return(
        self,
        pi_state: torch.FloatTensor,
        beta_state: torch.FloatTensor,
        item_index: torch.LongTensor,
        episodic_return: torch.FloatTensor,
    ) -> torch.FloatTensor:
        action_policy_prob = torch.exp(self.action_policy(pi_state, item_index))
        behavior_policy_prob = torch.exp(self.behavior_policy(beta_state, item_index))
        importance_weight = torch.div(action_policy_prob, behavior_policy_prob)
        return torch.mul(importance_weight, episodic_return)
