from typing import Sequence

import torch
import torch.nn as nn


class SoftmaxStochasticPolicy(nn.Module):
    def __init__(
        self,
        n_items: int,
        item_embedding_dim: int,
        adaptive_softmax: bool = False,
        state_vector_dim: int = None,
        softmax_cutoffs: Sequence = None,
        softmax_temperature: float = 1.0,
    ):
        super(SoftmaxStochasticPolicy, self).__init__()

        self.item_embeddings = nn.Embedding(
            num_embeddings=n_items,
            embedding_dim=item_embedding_dim,
        )

        self.adaptive_softmax = adaptive_softmax
        if self.adaptive_softmax is True:
            assert state_vector_dim, "State vector dimension should be provided."
            self.softmax = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=state_vector_dim + item_embedding_dim,
                n_classes=n_items,
                cutoffs=softmax_cutoffs,
            )
        else:
            self.softmax = nn.Softmax(dim=1)
            self.item_space = torch.arange(n_items)
            self.T = softmax_temperature

    def forward(
        self, state: torch.FloatTensor, item_index: torch.LongTensor
    ) -> torch.FloatTensor:
        batch_size = item_index.size(0)
        if self.adaptive_softmax is True:
            item_embedded = self.item_embeddings(item_index).view(batch_size, -1)
            log_item_prob = self.softmax(
                torch.cat((state, item_embedded), dim=1), item_index.squeeze()
            ).output
        else:
            assert state.size(-1) == self.item_embeddings.weight.size(
                -1
            ), "State & item embedding vector size should match."
            items_embedded = self.item_embeddings(self.item_space)
            logits = torch.stack(
                [torch.sum(s * items_embedded / self.T, dim=1) for s in state]
            )
            item_probs = self.softmax(logits)
            item_prob = torch.cat(
                [
                    item_probs[batch_idx][item_index[batch_idx]]
                    for batch_idx in range(batch_size)
                ]
            ).view(batch_size, -1)
            log_item_prob = torch.log(item_prob)

        return log_item_prob
