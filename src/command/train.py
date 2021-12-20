import time
from itertools import chain
from typing import List, Optional, Tuple, Union

import torch
from torch.optim import Adam
from tqdm import tqdm

from ..dataset.retailrocket import RetailrocketEpisodeLoader
from ..model.agent import TopKOfflineREINFORCE
from ..model.baseline import GRU4Rec
from ..model.policy import BehaviorPolicy


def train_GRU4Rec(
    model: Union[GRU4Rec, BehaviorPolicy],
    train_loader: RetailrocketEpisodeLoader,
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

            # 1. Build Logits
            if isinstance(model, BehaviorPolicy):
                state, lengths = model.struct_state(batch["pack_padded_histories"])
                logits = model.log_probs(state)
            else:
                logits, lengths = model(batch["pack_padded_histories"])

            # 2. Compute TOP1 Loss
            items_appeared = set(chain(*[ep for ep in batch["item_episodes"]]))
            losses = []
            for logit, length, item_episode in zip(
                logits, lengths, batch["item_episodes"]
            ):
                other_logits = []
                relevant_logit = []
                for i in range(length):
                    curr = item_episode[i + 1]
                    negative_samples = list(items_appeared - set(item_episode))
                    other_logits.append(logit[i, negative_samples].view(1, -1))
                    relevant_logit.append(logit[i, curr].view(1))
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

            if device.type == "cuda":
                torch.cuda.empty_cache()

        if debug is True:
            print(f"Epoch {epoch} Final loss: {loss.cpu().item()}")
        print("=" * 80)

    return train_loss_log


def train_agent(
    agent: TopKOfflineREINFORCE,
    train_loader: RetailrocketEpisodeLoader,
    n_epochs: Union[int, Tuple[int]],
    device: torch.device = torch.device("cpu"),
    debug: bool = False,
    behavior_policy_pretrained: bool = False,
) -> Optional[Tuple[List[float], List[float]]]:
    agent = agent.to(device)

    if isinstance(n_epochs, int):
        n_epochs_beta = n_epochs_pi = n_epochs
    elif isinstance(n_epochs, tuple):
        n_epochs_beta, n_epochs_pi = n_epochs
    else:
        raise TypeError(
            f"Unregistered {type(n_epochs)} type n_epochs entered.: {n_epochs}"
        )

    # 1. Train behavior policy first
    if behavior_policy_pretrained is False:
        agent.behavior_policy.train()
        behavior_policy_loss_log = train_GRU4Rec(
            model=agent.behavior_policy,
            train_loader=train_loader,
            n_epochs=n_epochs_beta,
            device=device,
            debug=debug,
        )

    # 2. Train action policy
    agent.train()
    agent.behavior_policy.eval()
    action_policy_loss_log = []
    for epoch in range(1, n_epochs_pi + 1):
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"\nEpoch {epoch} for action policy\nstarted at: {start_time}\n")

        for batch in tqdm(train_loader, desc="train"):
            batch = train_loader.to(batch=batch, device=device)

            # 1. Build States
            beta_state, _ = agent.behavior_policy.struct_state(
                batch["pack_padded_histories"]
            )

            pi_state, lengths = agent.pi_state_network(batch["pack_padded_histories"])

            # 2. Compute Policy Loss
            losses = []
            for (
                ep_pi_state,
                ep_beta_state,
                hist_len,
                ep_item_index,
                ep_return_at_t,
            ) in zip(
                pi_state,
                beta_state,
                lengths,
                batch["item_episodes"],
                batch["return_at_t"],
            ):
                ep_len = hist_len + 1
                episodic_loss = agent.episodic_loss(
                    pi_state=ep_pi_state[:hist_len, :],
                    beta_state=ep_beta_state[:hist_len, :],
                    item_index=torch.LongTensor(ep_item_index[1:ep_len]).view(
                        hist_len, 1
                    ),
                    episodic_return=torch.FloatTensor(ep_return_at_t[1:ep_len]).view(
                        hist_len, 1
                    ),
                )

                losses.append(episodic_loss.view(1))
            action_policy_loss = torch.cat(losses).mean()

            # 3. Gradient update agent w/ computed losses
            agent.action_policy_optimizer.zero_grad()
            action_policy_loss.backward()
            agent.action_policy_optimizer.step()

            if debug is True:
                action_policy_loss_log.append(action_policy_loss.cpu().item())

        if debug is True:
            print(f"Final action policy loss: {action_policy_loss.cpu().item()}")

        print("=" * 80)

    if debug is True:
        return action_policy_loss_log, behavior_policy_loss_log
