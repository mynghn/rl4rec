from typing import Sequence

import torch
import torch.nn as nn


class SoftmaxStochasticPolicy(nn.Module):
    def __init__(
        self,
        n_items: int,
        item_embedding_dim: int,
        adaptive_softmax: bool = False,
        softmax_cutoffs: Sequence = None,
        softmax_temperature: float = 1.0,
    ):
        super(SoftmaxStochasticPolicy, self).__init__()

        self.action_space = torch.arange(n_items)
        self.action_embeddings = nn.Embedding(
            num_embeddings=n_items,
            embedding_dim=item_embedding_dim,
        )

        self.adaptive_softmax = adaptive_softmax
        if self.adaptive_softmax is True:
            self.softmax = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=1,
                n_classes=n_items,
                cutoffs=softmax_cutoffs,
            )
        else:
            self.softmax = nn.Softmax(dim=1)
        self.T = softmax_temperature

    def forward(self, state: torch.FloatTensor) -> torch.FloatTensor:
        assert state.size(-1) == self.action_embeddings.weight.size(
            -1
        ), "State & item embedding vector size should match."

        items_embedded = self.action_embeddings(self.action_space)

        logits = torch.stack(
            [torch.sum(s * items_embedded / self.T, dim=1) for s in state]
        )
        if self.adaptive_softmax is True:
            item_probs = self.softmax(logits, self.action_space).output
        else:
            item_probs = self.softmax(logits)

        return item_probs
