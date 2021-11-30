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

        self.num_actions = self.action_policy.action_space.size(0)

        self.K = K

        self.weight_cap = weight_cap

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
        action: torch.LongTensor,
        episodic_return: torch.FloatTensor,
    ) -> torch.FloatTensor:
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

        return -torch.mean(
            _importance_weight * _lambda_K * episodic_return * log_action_policy_prob,
            dim=0,
        )

    def behavior_policy_loss(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(self.behavior_policy.softmax, nn.AdaptiveLogSoftmaxWithLoss):
            log_behavior_policy_prob = self.behavior_policy(state)[:, action]
        else:
            log_behavior_policy_prob = torch.log(self.behavior_policy(state)[:, action])

        batch_size = action.size(0)
        actual_action_prob = torch.zeros_like(log_behavior_policy_prob).index_put(
            indices=(torch.arange(batch_size), action.squeeze().long()),
            values=torch.ones(batch_size),
        )

        return self.kl_div_loss(log_behavior_policy_prob, actual_action_prob)

    def update_action_policy(self, action_policy_loss: torch.Tensor):
        self.action_policy_optimizer.zero_grad()
        action_policy_loss.backward()
        self.action_policy_optimizer.step()

    def update_behavior_policy(self, behavior_policy_loss: torch.Tensor):
        self.behavior_policy_optimizer.zero_grad()
        behavior_policy_loss.backward()
        self.behavior_policy_optimizer.step()
