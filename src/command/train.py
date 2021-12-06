import time
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from ..dataset.cikim19 import CIKIM19DataLoader
from ..model.agent import TopKOfflineREINFORCE


def train_agent(
    agent: TopKOfflineREINFORCE,
    train_loader: CIKIM19DataLoader,
    n_epochs: int,
    device: torch.device = torch.device("cpu"),
    debug: bool = True,
) -> Optional[Tuple[List[float], List[float]]]:
    action_policy_loss_log = []
    behavior_policy_loss_log = []
    for epoch in range(1, n_epochs + 1):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"\nEpoch {epoch} started at: {start_time}\n")

        agent = agent.to(device)
        agent.train()
        for batch_dict in tqdm(train_loader, desc="train"):
            batch_dict = train_loader.to(batch=batch_dict, device=device)

            # 1. Build State
            if (
                agent.state_network.user_feature_enabled
                and agent.state_network.item_feature_enabled
            ):
                state = agent.state_network(
                    user_history=batch_dict["user_history"],
                    user_feature_index=batch_dict.get("user_feature_index"),
                    item_feature_index=batch_dict.get("item_feature_index"),
                )
            elif agent.state_network.user_feature_enabled:
                state = agent.state_network(
                    user_history=batch_dict["user_history"],
                    user_feature_index=batch_dict.get("user_feature_index"),
                )
            elif agent.state_network.item_feature_enabled:
                state = agent.state_network(
                    user_history=batch_dict["user_history"],
                    item_feature_index=batch_dict.get("item_feature_index"),
                )
            else:
                state = agent.state_network(user_history=batch_dict["user_history"])

            # 2. Compute Policy Losses
            action_policy_loss = agent.action_policy_loss(
                state=state,
                episodic_return=batch_dict["return"],
                item_index=batch_dict["item_index"],
            )
            behavior_policy_loss = agent.behavior_policy_loss(
                state=state, item_index=batch_dict["item_index"]
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
