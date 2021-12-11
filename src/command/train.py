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
    agent = agent.to(device)
    agent.train()

    # 1. Train behavior policy first
    behavior_policy_loss_log = []
    for epoch in range(1, n_epochs + 1):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"\nEpoch {epoch} for behavior policy train\nstarted at: {start_time}\n")

        for batch in tqdm(train_loader, desc="train"):
            batch = train_loader.to(batch=batch, device=device)

            # 1. Build State
            beta_state = agent.beta_state_network(
                user_history=batch["user_history"],
                user_feature_index=batch.get("user_feature_index"),
                item_feature_index=batch.get("item_feature_index"),
            )

            # 2. Compute Policy Loss
            behavior_policy_loss = agent.behavior_policy_loss(
                beta_state=beta_state, item_index=batch["item_index"]
            )

            # 3. Gradient update agent w/ computed losses
            agent.update_behavior_policy(behavior_policy_loss=behavior_policy_loss)

            if debug is True:
                behavior_policy_loss_log.append(behavior_policy_loss.cpu().item())

        if debug is True:
            print(f"Final behavior policy loss: {behavior_policy_loss.cpu().item()}")

    # 2. Train action policy
    action_policy_loss_log = []
    agent.beta_state_network.eval()
    for epoch in range(1, n_epochs + 1):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"\nEpoch {epoch} for action policy train\nstarted at: {start_time}\n")

        for batch in tqdm(train_loader, desc="train"):
            batch = train_loader.to(batch=batch, device=device)

            # 1. Build States
            beta_state = agent.beta_state_network(
                user_history=batch["user_history"],
                user_feature_index=batch.get("user_feature_index"),
                item_feature_index=batch.get("item_feature_index"),
            )
            pi_state = agent.pi_state_network(
                user_history=batch["user_history"],
                user_feature_index=batch.get("user_feature_index"),
                item_feature_index=batch.get("item_feature_index"),
            )

            # 2. Compute Policy Loss
            action_policy_loss = agent.action_policy_loss(
                pi_state=pi_state,
                beta_state=beta_state,
                item_index=batch["item_index"],
                episodic_return=batch["return"],
            )

            # 3. Gradient update agent w/ computed losses
            agent.update_action_policy(action_policy_loss=action_policy_loss)

            if debug is True:
                action_policy_loss_log.append(action_policy_loss.cpu().item())

        if debug is True:
            print(f"Final action policy loss: {action_policy_loss.cpu().item()}")

        print("=" * 80)

    return action_policy_loss_log, behavior_policy_loss_log
