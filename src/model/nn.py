import torch
import torch.nn as nn


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
            num_embeddings=num_items,
            embedding_dim=item_embedding_size,
        )
        self.rnn = nn.GRU(
            input_size=item_embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bias=True,
            batch_first=True,
        )

    def forward(self, user_history: torch.Tensor) -> torch.Tensor:
        user_history_embedded = self.item_embeddings(user_history)
        next_state, _ = self.rnn(input=user_history_embedded)
        return next_state
