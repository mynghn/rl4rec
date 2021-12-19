import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class StateTransitionNetwork(nn.Module):
    def __init__(
        self,
        n_items: int,
        hidden_size: int,
        num_layers: int,
        dropout: int,
        user_action_embedding_dim: int = -1,
        n_actions: int = None,
        padding_singal: int = None,
    ):
        super(StateTransitionNetwork, self).__init__()
        self.padding_signal = padding_singal

        self.n_items = n_items

        if user_action_embedding_dim > 0:
            self.user_action_embeddings = nn.Embedding(
                num_embeddings=n_items * n_actions + 1,
                embedding_dim=user_action_embedding_dim,
                padding_idx=-1,
            )
            rnn_input_size = user_action_embedding_dim
        else:
            rnn_input_size = n_items

        self.rnn = nn.GRU(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bias=True,
            batch_first=True,
        )

    def forward(self, pack_padded_histories: PackedSequence) -> torch.FloatTensor:
        packed_output, _ = self.rnn(input=pack_padded_histories)
        output, lengths = pad_packed_sequence(sequence=packed_output, batch_first=True)

        next_state = torch.stack(
            [
                output[batch_idx, lengths[batch_idx] - 1, :]
                for batch_idx in range(output.size(0))
            ]
        )

        return next_state
