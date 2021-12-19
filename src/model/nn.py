import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..dataset.custom_typings import PaddedNSortedUserHistoryBatch


class StateTransitionNetwork(nn.Module):
    def __init__(
        self,
        n_user_actions: int,
        user_action_embedding_dim: int,
        hidden_size: int,
        padding_singal: int,
        num_layers: int,
        dropout: int,
    ):
        super(StateTransitionNetwork, self).__init__()
        self.padding_signal = padding_singal
        self.user_action_embeddings = nn.Embedding(
            num_embeddings=n_user_actions + 1,
            embedding_dim=user_action_embedding_dim,
            padding_idx=-1,
        )
        self.rnn = nn.GRU(
            input_size=user_action_embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bias=True,
            batch_first=True,
        )

    def forward(self, user_history: PaddedNSortedUserHistoryBatch) -> torch.FloatTensor:
        padding_idx_replaced = user_history.data.masked_fill(
            user_history.data == self.padding_signal,
            self.user_action_embeddings.padding_idx,
        )
        user_history_embedded = self.user_action_embeddings(padding_idx_replaced)
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
