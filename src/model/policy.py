from typing import Sequence

import torch
import torch.nn as nn


class SoftmaxStochasticPolicy(nn.Module):
    def __init__(
        self,
        num_actions: int,
        action_embedding_size: int,
        adaptive_softmax: bool = False,
        softmax_cutoffs: Sequence = None,
        softmax_temperature: float = 1.0,
    ):
        super(SoftmaxStochasticPolicy, self).__init__()

        self.action_space = torch.range(num_actions)
        self.action_embeddings = nn.Embedding(
            num_embeddings=num_actions,
            embedding_dim=action_embedding_size,
        )

        self.adaptive_softmax = adaptive_softmax
        if self.adaptive_softmax is True:
            self.softmax = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=1,
                n_classes=num_actions,
                cutoffs=softmax_cutoffs,
            )
        else:
            self.softmax = nn.Softmax(dim=1)
        self.T = softmax_temperature

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        assert state.size(1) == self.action_embeddings.weight.size(
            1
        ), "User state & action embedding should have same size of dimensions."

        actions_embedded = self.action_embeddings(self.action_space)

        logits = torch.stack(
            torch.sum(s * actions_embedded / self.T, dim=1) for s in state
        )
        if self.adaptive_softmax is True:
            action_probs = self.softmax(logits, self.action_space).output
        else:
            action_probs = self.softmax(logits)

        return action_probs
