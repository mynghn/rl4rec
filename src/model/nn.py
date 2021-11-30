import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..dataset.dataset import PaddedNSortedUserHistoryBatch


class StateTransitionNetwork(nn.Module):
    def __init__(
        self,
        num_items: int,
        item_embedding_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: int = 0,
    ):
        super(StateTransitionNetwork, self).__init__()
        self.item_embeddings = nn.Embedding(
            num_embeddings=num_items + 1,
            embedding_dim=item_embedding_size,
            padding_idx=0,
        )
        self.rnn = nn.GRU(
            input_size=item_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bias=True,
            batch_first=True,
        )

    def forward(self, user_history: PaddedNSortedUserHistoryBatch) -> torch.FloatTensor:
        user_history_embedded = self.item_embeddings(user_history.data)
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
