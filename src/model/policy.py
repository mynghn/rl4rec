from typing import Sequence

import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence

from ..model.baseline import GRU4Rec


class SoftmaxStochasticPolicyHead(nn.Module):
    def __init__(
        self,
        n_items: int,
        adaptive_softmax: bool,
        state_vector_dim: int = None,
        softmax_cutoffs: Sequence = None,
        item_embedding_dim: int = None,
        softmax_temperature: float = 1.0,
    ):
        super(SoftmaxStochasticPolicyHead, self).__init__()

        self.adaptive_softmax = adaptive_softmax
        if self.adaptive_softmax is True:
            assert (
                state_vector_dim is not None and softmax_cutoffs is not None
            ), "Both state vector dimension & class cutoffs should be provided for adaptive softmax."

            self.softmax = nn.AdaptiveLogSoftmaxWithLoss(
                in_features=state_vector_dim,
                n_classes=n_items,
                cutoffs=softmax_cutoffs,
            )
        else:
            assert (
                item_embedding_dim is not None
            ), "Item embedding dimension should be provided for full softmax."

            self.softmax = nn.Softmax(dim=1)
            self.item_space = torch.arange(n_items)
            self.item_embeddings = nn.Embedding(
                num_embeddings=n_items,
                embedding_dim=item_embedding_dim,
            )
            self.T = softmax_temperature

    def forward(
        self, state: torch.FloatTensor, item_index: torch.LongTensor
    ) -> torch.FloatTensor:
        batch_size = item_index.size(0)
        if self.adaptive_softmax is True:
            log_item_prob = self.softmax(state, item_index.view(batch_size)).output
        else:
            log_items_prob = self.log_probs(state)
            log_item_prob = torch.cat(
                [
                    log_items_prob[batch_idx][item_index[batch_idx]]
                    for batch_idx in range(batch_size)
                ]
            )

        return log_item_prob.view(batch_size, -1)

    def log_probs(self, state: torch.FloatTensor) -> torch.FloatTensor:
        if self.adaptive_softmax is True:
            log_items_prob = self.softmax.log_prob(state)
        else:
            assert state.size(-1) == self.item_embeddings.weight.size(
                -1
            ), "State & item embedding vector size should match."
            items_embedded = self.item_embeddings(self.item_space)
            logits = torch.stack(
                [torch.sum(s * items_embedded / self.T, dim=1) for s in state]
            )
            log_items_prob = torch.log(self.softmax(logits) + 1e-8)

        return log_items_prob


class BehaviorPolicy(GRU4Rec):
    def __init__(
        self,
        n_items: int,
        hidden_size: int,
        n_gru_layers: int = 1,
        dropout: int = 0.4,
        user_action_embedding_dim: int = -1,
        n_actions: int = None,
        padding_signal: int = None,
    ):
        super().__init__(
            n_items,
            hidden_size,
            n_gru_layers=n_gru_layers,
            dropout=dropout,
            user_action_embedding_dim=user_action_embedding_dim,
            n_actions=n_actions,
            padding_signal=padding_signal,
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, state: torch.FloatTensor, item_index: torch.LongTensor
    ) -> torch.FloatTensor:
        batch_size = item_index.size(0)
        log_items_prob = self.log_probs(state)
        log_item_prob = torch.cat(
            [
                log_items_prob[batch_idx][item_index[batch_idx]]
                for batch_idx in range(batch_size)
            ]
        )

        return log_item_prob.view(batch_size, -1)

    def log_probs(self, state: torch.FloatTensor) -> torch.FloatTensor:
        logits = self.output_layer(state)
        return torch.log(self.softmax(logits) + 1e-8)

    def struct_state(self, pack_padded_histories: PackedSequence) -> torch.FloatTensor:
        return self.gru_layer(pack_padded_histories)
