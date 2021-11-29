import time
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from ..dataset.dataset import UserItemEpisodeLoader
from ..model.agent import TopKOfflineREINFORCE


def train_agent(
    agent: TopKOfflineREINFORCE,
    dataloader: UserItemEpisodeLoader,
    n_epochs: int,
    device: torch.device = torch.device("cpu"),
    debug: bool = True,
) -> Optional[Tuple[List[float], List[float]]]:
    action_policy_loss_log = []
    behavior_policy_loss_log = []
    for epoch in range(1, n_epochs + 1):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"\nEpoch {epoch} started at: {start_time}\n")

        agent.train()
        for prev_item_sequence, action, episodic_return in tqdm(
            dataloader, desc="train"
        ):
            if device.type != "cpu":
                prev_item_sequence = prev_item_sequence.to(device)
                episodic_return = episodic_return.to(device)

            user_state = torch.Tensor(
                [agent.state_network(seq) for seq in prev_item_sequence]
            )

            action_policy_loss = agent.action_policy_loss(
                state=user_state, action=action
            )
            agent.update_action_policy(action_policy_loss=action_policy_loss)

            behavior_policy_loss = agent.behavior_policy_loss(
                state=user_state, action=action
            )
            agent.update_behavior_policy(behavior_policy_loss=behavior_policy_loss)

            if debug:
                action_policy_loss_log.append(action_policy_loss)
                behavior_policy_loss_log.append(behavior_policy_loss)

        if debug:
            print(f"1. Epoch {epoch} final action policy loss: {action_policy_loss}")
            print(
                f"2. Epoch {epoch} final behavior policy loss: {behavior_policy_loss}"
            )

        print("=" * 80)

    return action_policy_loss_log, behavior_policy_loss_log
