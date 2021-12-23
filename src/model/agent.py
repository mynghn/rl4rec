from typing import Tuple

import torch
import torch.nn as nn

from .nn import StateTransitionNetwork
from .policy import BehaviorPolicy, SoftmaxStochasticPolicyHead


class TopKOfflineREINFORCE(nn.Module):
    def __init__(
        self,
        pi_state_network: StateTransitionNetwork,
        action_policy_head: SoftmaxStochasticPolicyHead,
        behavior_policy: BehaviorPolicy,
        weight_cap: float,
        K: int,
    ):
        super(TopKOfflineREINFORCE, self).__init__()

        self.pi_state_network = pi_state_network
        self.action_policy_head = action_policy_head
        self.behavior_policy = behavior_policy

        self.weight_cap = weight_cap
        self.K = K

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

    def episodic_loss(
        self,
        pi_state: torch.FloatTensor,
        beta_state: torch.FloatTensor,
        item_index: torch.LongTensor,
        return_at_t: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        T: Length of the full episode from timestep t to (t+T-1)

        pi_state (T-1, N1): N1-dimensional state representations of action policy at timestmp (t+1) ~ (t+T-1)
        beta_state (T-1, N2): N2-dimensional state representations of behavior policy at timestmp (t+1) ~ (t+T-1)
        item_index (T-1, 1): Item indices meaning "action" at timestep (t+1) ~ (t+T-1)
        return_at_t (T-1, 1): Cumulative returns at timestmp (t+1) ~ (t+T-1)
        """
        log_action_policy_prob = self.action_policy_head(pi_state, item_index)
        action_policy_prob = torch.exp(log_action_policy_prob)

        behavior_policy_prob = torch.exp(
            self.behavior_policy(beta_state.detach(), item_index)
        )

        _lambda_K = self._compute_lambda_K(policy_prob=action_policy_prob)
        _importance_weight = self._compute_importance_weight(
            action_policy_prob=action_policy_prob,
            behavior_policy_prob=behavior_policy_prob,
        )

        return -torch.sum(
            _importance_weight * _lambda_K * return_at_t * log_action_policy_prob,
            dim=0,
        )

    def get_top_recommendations(
        self, state: torch.FloatTensor, M: int
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        if self.action_policy_head.adaptive_softmax is True:
            action_policy_probs = torch.exp(self.action_policy_head.log_probs(state))
        else:
            assert state.size(
                -1
            ) == self.action_policy_head.item_embeddings.weight.size(
                -1
            ), "State & item embedding vector size should match."
            items_embedded = self.action_policy_head.item_embeddings(
                self.action_policy_head.item_space
            )
            logits = torch.stack(
                [
                    torch.sum(s * items_embedded / self.action_policy_head.T, dim=1)
                    for s in state.detach()
                ]
            )
            action_policy_probs = self.action_policy_head.softmax(logits)

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
        if self.action_policy_head.adaptive_softmax is True:
            return super().to(device)
        else:
            on_device = super().to(device)
            on_device.action_policy_head.item_space = (
                on_device.action_policy_head.item_space.to(device)
            )
            return on_device

    def get_corrected_episodic_return(
        self,
        pi_state: torch.FloatTensor,
        beta_state: torch.FloatTensor,
        item_index: torch.LongTensor,
        return_at_t: torch.FloatTensor,
    ) -> float:
        """
        T: Length of the full episode from timestep t to (t+T-1)

        pi_state (T-1, N1): N1-dimensional state representations of action policy at timestmp (t+1) ~ (t+T-1)
        beta_state (T-1, N2): N2-dimensional state representations of behavior policy at timestmp (t+1) ~ (t+T-1)
        item_index (T-1, 1): Item indices meaning "action" at timestep (t+1) ~ (t+T-1)
        return_at_t (T-1, 1): Cumulative returns at timestmp (t+1) ~ (t+T-1)
        """
        action_policy_log_probs_in_episode = self.action_policy_head(
            pi_state, item_index
        ).view(-1)
        behavior_policy_log_probs_in_episode = self.behavior_policy(
            beta_state, item_index
        ).view(-1)

        episodic_return_cumulated = 0.0
        hist_len = item_index.size(0)
        for t in range(hist_len):
            importance_weight = torch.exp(
                action_policy_log_probs_in_episode[t]
                - behavior_policy_log_probs_in_episode[t]
            ).squeeze()
            episodic_return_cumulated += (
                importance_weight.cpu().item() * return_at_t[t, 0].cpu().item()
            )

        return episodic_return_cumulated / hist_len
