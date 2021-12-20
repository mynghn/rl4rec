import time
from itertools import chain
from typing import List, Optional, Tuple, Union

import torch
from torch.optim import Adam
from tqdm import tqdm

from ..dataset.retailrocket import Retailrocket4GRU4RecLoader, RetailrocketDataLoader
from ..model.agent import TopKOfflineREINFORCE
from ..model.baseline import GRU4Rec


def train_agent(
    agent: TopKOfflineREINFORCE,
    train_loader: RetailrocketDataLoader,
    n_epochs: Union[int, Tuple[int]],
    device: torch.device = torch.device("cpu"),
    debug: bool = False,
) -> Optional[Tuple[List[float], List[float]]]:
    agent = agent.to(device)
    agent.train()

    if isinstance(n_epochs, int):
        n_epochs_beta = n_epochs_pi = n_epochs
    elif isinstance(n_epochs, tuple):
        n_epochs_beta, n_epochs_pi = n_epochs
    else:
        raise TypeError(
            f"Unregistered {type(n_epochs)} type n_epochs entered.: {n_epochs}"
        )

    # 1. Train behavior policy first
    behavior_policy_loss_log = []
    for epoch in range(1, n_epochs_beta + 1):
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
    agent.behavior_policy.eval()
    for epoch in range(1, n_epochs_pi + 1):
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


def train_GRU4Rec(
    model: GRU4Rec,
    train_loader: Retailrocket4GRU4RecLoader,
    n_epochs: int,
    device: torch.device = torch.device("cpu"),
    debug: bool = False,
) -> Optional[Tuple[List[float], List[float]]]:
    model = model.to(device)
    model.train()
    optimizer = Adam(model.parameters())

    train_loss_log = []
    for epoch in range(1, n_epochs + 1):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"\nEpoch {epoch} started at: {start_time}\n")

        for batch in tqdm(train_loader, desc="train"):
            batch = train_loader.to(batch=batch, device=device)

            # 1. Build State
            batch_logits, lengths = model(batch["pack_padded_histories"])

            # 2. Compute TOP1 Loss
            items_appeared = set(chain(*[ep for ep in batch["item_episodes"]]))
            losses = []
            for logits, length, item_episode in zip(
                batch_logits, lengths, batch["item_episodes"]
            ):
                other_logits = []
                relevant_logit = []
                for i in range(length):
                    curr = item_episode[i + 1]
                    negative_samples = list(items_appeared - set(item_episode))
                    other_logits.append(logits[i, negative_samples].view(1, -1))
                    relevant_logit.append(logits[i, curr].view(1))
                episode_loss = model.top1_loss(
                    torch.cat(other_logits), torch.cat(relevant_logit)
                )

                losses.append(episode_loss.sum().view(1))
            loss = torch.cat(losses).mean()

            # 3. Gradient update agent w/ computed losses
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if debug is True:
                train_loss_log.append(loss.cpu().item())

        if debug is True:
            print(f"Epoch {epoch} Final loss: {loss.cpu().item()}")
        print("=" * 80)

    return train_loss_log
