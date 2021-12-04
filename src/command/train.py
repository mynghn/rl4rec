import time
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.agent import TopKOfflineREINFORCE


def train_agent(
    agent: TopKOfflineREINFORCE,
    train_loader: DataLoader,
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
        for batch_dict in tqdm(train_loader, desc="train"):
            if device.type != "cpu":
                batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

                agent = agent.to(device)
                agent.action_policy.action_space = agent.action_policy.action_space.to(
                    device
                )
                agent.behavior_policy.action_space = (
                    agent.behavior_policy.action_space.to(device)
                )

            # 1. Build State
            state = agent.state_network(
                user_history=batch_dict["user_history"],
                user_feature_index=batch_dict.get("user_feature_index"),
                item_feature_index=batch_dict.get("item_feature_index"),
            )

            # 2. Compute Policy Losses
            action_policy_loss = agent.action_policy_loss(
                state=state,
                action_index=batch_dict["action_index"],
                episodic_return=batch_dict["return"],
            )
            behavior_policy_loss = agent.behavior_policy_loss(
                state=state, action_index=batch_dict["action_index"]
            )

            # 3. Gradient update agent w/ computed losses
            agent.update(
                action_policy_loss=action_policy_loss,
                behavior_policy_loss=behavior_policy_loss,
            )

            if debug:
                action_policy_loss_log.append(action_policy_loss.cpu().item())
                behavior_policy_loss_log.append(behavior_policy_loss.cpu().item())

        if debug:
            print(
                f"1. Epoch {epoch}'s final action policy loss: {action_policy_loss.cpu().item()}"
            )
            print(
                f"2. Epoch {epoch}'s final behavior policy loss: {behavior_policy_loss.cpu().item()}"
            )

        print("=" * 80)

    return action_policy_loss_log, behavior_policy_loss_log
