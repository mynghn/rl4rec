import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..dataset.custom_typings import PaddedNSortedUserHistoryBatch


class StateTransitionNetwork(nn.Module):
    def __init__(
        self,
        n_actions: int,
        action_embedding_size: int,
        hidden_size: int,
        padding_singal: int,
        num_layers: int = 1,
        dropout: int = 0,
        user_feature: bool = False,
        user_feature_dim: int = None,
        n_user_features: int = None,
        item_feature: bool = False,
        item_feature_dim: int = None,
        n_item_features: int = None,
    ):
        super(StateTransitionNetwork, self).__init__()
        self.item_embeddings = nn.Embedding(
            num_embeddings=n_actions + 1,
            embedding_dim=action_embedding_size,
            padding_idx=-1,
        )
        self.rnn = nn.GRU(
            input_size=action_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bias=True,
            batch_first=True,
        )

        self.padding_signal = padding_singal

        self.user_feature_enabled = user_feature
        if self.user_feature_enabled is True:
            self.user_feature_embeddings = nn.Embedding(
                num_embeddings=n_user_features,
                embedding_dim=user_feature_dim,
            )
        self.item_feature_enabled = item_feature
        if self.item_feature_enabled is True:
            self.item_feature_embeddings = nn.Embedding(
                num_embeddings=n_item_features,
                embedding_dim=item_feature_dim,
            )

    def forward(self, *args, **kargs) -> torch.FloatTensor:
        if self.user_feature_enabled and self.item_feature_enabled:
            self.forward_w_user_item_feature(*args, **kargs)
        elif self.user_feature_enabled:
            self.forward_w_user_feature(*args, **kargs)
        elif self.item_feature_enabled:
            self.forward_w_item_feature(*args, **kargs)
        else:
            self.forward_only_user_history(*args, **kargs)

    def forward_only_user_history(
        self, user_history: PaddedNSortedUserHistoryBatch
    ) -> torch.FloatTensor:
        padding_idx_replaced = user_history.data.masked_fill(
            user_history.data == self.padding_signal, self.item_embeddings.padding_idx
        )
        user_history_embedded = self.item_embeddings(padding_idx_replaced)
        user_history_packedNembedded = pack_padded_sequence(
            input=user_history_embedded,
            lengths=user_history.lengths,
            batch_first=True,
        )

        packed_output, _ = self.rnn(input=user_history_packedNembedded)
        output, lengths = pad_packed_sequence(sequence=packed_output, batch_first=True)

        next_state = torch.stack(
            [
                output[batch_idx, lengths[batch_idx] - 1, :]
                for batch_idx in range(output.size(0))
            ]
        )

        return next_state

    def forward_w_user_item_feature(
        self,
        user_history: PaddedNSortedUserHistoryBatch,
        user_feature_index: torch.LongTensor,
        item_feature_index: torch.LongTensor,
    ) -> torch.FloatTensor:
        rnn_output = self.forward_only_user_history(user_history)
        user_feature_embedded = self.user_feature_embeddings(user_feature_index)
        item_feature_embedded = self.item_feature_embeddings(item_feature_index)

        return torch.cat(
            [rnn_output, user_feature_embedded, item_feature_embedded], dim=1
        )

    def forward_w_user_feature(
        self,
        user_history: PaddedNSortedUserHistoryBatch,
        user_feature_index: torch.LongTensor,
    ) -> torch.FloatTensor:
        rnn_output = self.forward_only_user_history(user_history)
        user_feature_embedded = self.user_feature_embeddings(user_feature_index)

        return torch.cat([rnn_output, user_feature_embedded], dim=1)

    def forward_w_item_feature(
        self,
        user_history: PaddedNSortedUserHistoryBatch,
        item_feature_index: torch.LongTensor,
    ) -> torch.FloatTensor:
        rnn_output = self.forward_only_user_history(user_history)
        item_feature_embedded = self.item_feature_embeddings(item_feature_index)

        return torch.cat([rnn_output, item_feature_embedded], dim=1)
