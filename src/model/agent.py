import torch
import torch.nn as nn

from .nn import StateTransitionNetwork
from .policy import SoftmaxStochasticPolicy


class TopKOfflineREINFORCE(nn.Module):
    def __init__(
        self,
        state_network: StateTransitionNetwork,
        action_policy: SoftmaxStochasticPolicy,
        behavior_policy: SoftmaxStochasticPolicy,
        K: int,
        weight_cap: float,
    ):
        super(TopKOfflineREINFORCE, self).__init__()

        self.state_network = state_network
        self.action_policy = action_policy
        self.behavior_policy = behavior_policy

        self.num_actions = self.action_policy.action_space.size(0)

        self.K = K

        self.weight_cap = weight_cap

    def _compute_lambda_K(self, policy_prob: torch.Tensor) -> torch.Tensor:
        return self.K * ((1 - policy_prob) ** (self.K - 1))

    def _compute_importance_weight(
        self, action_policy_prob: torch.Tensor, behavior_policy_prob: torch.Tensor
    ) -> torch.Tensor:
        weight = torch.div(action_policy_prob, behavior_policy_prob)
        return torch.minimum(
            weight, torch.mul(torch.ones_like(weight), self.weight_cap)
        )

    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        episodic_return: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(self.action_policy.softmax, nn.AdaptiveLogSoftmaxWithLoss):
            log_action_policy_prob = self.action_policy(state)[:, action]
            action_policy_prob = torch.exp(log_action_policy_prob)
        else:
            action_policy_prob = self.action_policy(state)[:, action]
            log_action_policy_prob = torch.log(action_policy_prob)

        if isinstance(self.behavior_policy.softmax, nn.AdaptiveLogSoftmaxWithLoss):
            behavior_policy_prob = torch.exp(self.action_policy(state)[:, action])
        else:
            behavior_policy_prob = self.action_policy(state)[:, action]

        _lambda_K = self._compute_lambda_K(policy_prob=action_policy_prob)
        _importance_weight = self._compute_importance_weight(
            action_policy_prob=action_policy_prob,
            behavior_policy_prob=behavior_policy_prob,
        )

        batch_loss = (
            _importance_weight * _lambda_K * episodic_return * log_action_policy_prob
        )

        return torch.mean(batch_loss, dim=0)
