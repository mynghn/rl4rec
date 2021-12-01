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
                prev_item_sequence.data = prev_item_sequence.data.to(device)
                action = action.to(device)
                episodic_return = episodic_return.to(device)

                agent = agent.to(device)
                agent.action_policy.action_space = agent.action_policy.action_space.to(
                    device
                )
                agent.behavior_policy.action_space = (
                    agent.behavior_policy.action_space.to(device)
                )

            user_state = agent.state_network(prev_item_sequence)

            action_policy_loss = agent.action_policy_loss(
                state=user_state, action=action, episodic_return=episodic_return
            )
            behavior_policy_loss = agent.behavior_policy_loss(
                state=user_state, action=action
            )

            agent.update(
                action_policy_loss=action_policy_loss,
                behavior_policy_loss=behavior_policy_loss,
            )

            if debug:
                action_policy_loss_log.append(action_policy_loss.cpu().item())
                behavior_policy_loss_log.append(behavior_policy_loss.cpu().item())

            if device.type == "cuda":
                torch.cuda.empty_cache()

        if debug:
            print(f"1. Epoch {epoch}'s final action policy loss: {action_policy_loss}")
            print(
                f"2. Epoch {epoch}'s final behavior policy loss: {behavior_policy_loss}"
            )

        print("=" * 80)

    return action_policy_loss_log, behavior_policy_loss_log
