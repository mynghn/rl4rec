import os
from math import ceil
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.nn.functional import one_hot
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from .custom_typings import PaddedNSortedEpisodeBatch


class RetailrocketEpisodeDataset(Dataset):
    events_fetching_dtypes = {
        "visitorid": str,
        "itemid": str,
        "event": str,
        "timestamp": int,
    }
    item_properties_fetching_dtypes = {
        "itemid": str,
        "property": str,
        "value": str,
        "timestamp": int,
    }

    reward_map = {"view": 1.0, "addtocart": 2.0, "transaction": 4.0}

    def __init__(
        self,
        data_path: str,
        train: bool,
        split_ratio: float,
        interval_hours_in_episode: int = 24,
        item_space_size: int = None,
        category_id: str = None,
        train_item_index_map: pd.DataFrame = None,
    ):
        self.data_path = data_path

        self.train = train
        self.split_ratio = split_ratio

        self.category_id = category_id

        self.interval_hours_in_episode = interval_hours_in_episode
        self.item_space_size = item_space_size

        # 0. Fetch Raw data
        self.events_df = self._get_events_df()
        self.item_category_df = self._get_item_category_df()

        # 1. Build event logs
        self.logs = self._build_event_logs()

        # 2. Build item index map
        if self.train is True:
            self.item_index_map = self._build_item_index_map()
        else:
            assert (
                train_item_index_map is not None
            ), "Item-index mapping from train sequence should be provided for evaluation dataset."
            self.item_index_map = train_item_index_map

        # 3. Build final data
        self.episodes = self._build_episodes()

    def _get_events_df(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.data_path, "events.csv"),
            header=0,
            dtype=self.events_fetching_dtypes,
        )
        df.drop("transactionid", axis=1, inplace=True)
        return df

    def _get_item_category_df(self) -> pd.DataFrame:
        item_properties_part1 = pd.read_csv(
            os.path.join(self.data_path, "item_properties_part1.csv"),
            header=0,
            dtype=self.item_properties_fetching_dtypes,
        )
        item_properties_part2 = pd.read_csv(
            os.path.join(self.data_path, "item_properties_part2.csv"),
            header=0,
            dtype=self.item_properties_fetching_dtypes,
        )

        df = pd.concat([item_properties_part1, item_properties_part2])
        df = df[df.property == "categoryid"]
        df.sort_values(by="timestamp", ascending=False, inplace=True)
        df.drop_duplicates("itemid", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename(columns={"value": "categoryid"}, inplace=True)
        df.drop(["property", "timestamp"], axis=1, inplace=True)

        return df

    def _build_event_logs(self) -> pd.DataFrame:
        # 1. Merge & Collect only events with items in specific category
        preprocessed = self.events_df.merge(
            self.item_category_df, on="itemid", how="left"
        )
        if self.category_id:
            preprocessed = preprocessed[preprocessed.categoryid == self.category_id]

        # 2. Split data
        preprocessed.drop_duplicates(inplace=True)
        preprocessed.sort_values(by="timestamp", inplace=True)
        preprocessed.reset_index(drop=True, inplace=True)
        n_records = preprocessed.shape[0]
        if self.train is True:
            n_train = ceil(n_records * self.split_ratio)
            preprocessed = preprocessed.iloc[:n_train]
        else:
            n_train = ceil(n_records * (1 - self.split_ratio))
            preprocessed = preprocessed.iloc[n_train:]
            preprocessed.reset_index(drop=True, inplace=True)

        # 3. Reduce item space size
        if self.item_space_size:
            cnt_by_item = preprocessed.groupby("itemid")[["timestamp"]].count()
            cnt_by_item.sort_values(by="timestamp", ascending=False, inplace=True)
            popular_items = cnt_by_item.index.to_numpy()[: self.item_space_size]
            preprocessed = preprocessed[preprocessed.itemid.isin(popular_items)]

        # 4. Assign reward values
        preprocessed["reward"] = preprocessed.event.map(self.reward_map)

        return preprocessed

    def _build_item_index_map(self) -> pd.DataFrame:
        df = self.logs[["itemid"]].copy()
        df.drop_duplicates(inplace=True)
        df.sort_values(by="itemid", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename_axis("item_index", inplace=True)
        df.reset_index(drop=False, inplace=True)

        return df

    def _label_episode(self, timestamps: List[int]) -> List[int]:
        label = 0
        episode_labels = [label]
        for idx in range(1, len(timestamps)):
            if (
                timestamps[idx] - timestamps[idx - 1]
                > self.interval_hours_in_episode * 60 * 60 * 1000
            ):
                label += 1
            episode_labels.append(label)
        return episode_labels

    def _build_episodes(
        self,
    ) -> Tuple[
        pd.DataFrame,
        Dict[str, List[int]],
        Dict[str, List[float]],
        Dict[str, List[int]],
    ]:
        # 1. Merge indices
        df = self.logs.merge(self.item_index_map, on="itemid", how="inner")

        # 2. Assign event index within user history
        df.sort_values(by=["visitorid", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        groupedby_user = df.groupby(by="visitorid")
        df["event_index_in_user_history"] = groupedby_user.cumcount()

        # 3. Build lifetime sequence books by users
        lifetime_timestamps_book = groupedby_user["timestamp"].apply(list).to_dict()
        lifetime_items_book = groupedby_user["item_index"].apply(list).to_dict()
        lifetime_rewards_book = groupedby_user["reward"].apply(list).to_dict()

        # 4. Build episodes book
        episode_labels_book = {
            user_id: self._label_episode(timestamps)
            for user_id, timestamps in lifetime_timestamps_book.items()
        }
        episodes = []
        for user_id, episode_labels in episode_labels_book.items():
            item_seq = lifetime_items_book[user_id]
            reward_seq = lifetime_rewards_book[user_id]

            n_episodes = max(episode_labels) + 1
            for label in range(n_episodes):
                if episode_labels.count(label) > 1:
                    start = episode_labels.index(label)
                    if label + 1 < n_episodes:
                        end = episode_labels.index(label + 1)
                    else:
                        end = len(episode_labels)

                    item_episode = item_seq[start:end]
                    reward_episode = reward_seq[start:end]
                    episodes.append((user_id, item_episode, reward_episode))

        return np.array(episodes, dtype=object)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, idx: Any) -> List[Any]:
        return self.episodes[idx]


class RetailrocketDataLoader(DataLoader):
    padding_signal = -1

    def __init__(self, train: bool, *args, **kargs):
        self.train = train
        if self.train is True:
            super().__init__(collate_fn=self.train_collate_func, *args, **kargs)
        else:
            super().__init__(collate_fn=self.eval_collate_func, *args, **kargs)

    def pad_sequence(
        self, user_history: Sequence[Sequence[int]]
    ) -> Tuple[torch.LongTensor, torch.LongTensor]:
        lengths = torch.LongTensor([len(seq) for seq in user_history])
        max_length = lengths.max()
        padded = torch.stack(
            [
                torch.cat(
                    [
                        torch.LongTensor(item_seq),
                        torch.zeros(max_length - len(item_seq)) + self.padding_signal,
                    ]
                ).long()
                for item_seq in user_history
            ]
        )
        return padded, lengths

    def train_collate_func(
        self, batch: List[np.ndarray]
    ) -> Dict[
        str,
        Union[
            PaddedNSortedEpisodeBatch,
            torch.LongTensor,
            torch.FloatTensor,
        ],
    ]:
        batch_size = len(batch)
        user_history, item_index, _return = tuple(np.array(batch, dtype=object).T)

        padded_user_history, lengths = self.pad_sequence(user_history)
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        return {
            "user_history": PaddedNSortedEpisodeBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            "item_index": torch.from_numpy(
                item_index[sorted_idx].astype(np.int64)
            ).view(batch_size, -1),
            "return": torch.from_numpy(_return[sorted_idx].astype(np.float32)).view(
                batch_size, -1
            ),
        }

    def eval_collate_func(
        self, batch: List[np.ndarray]
    ) -> Dict[
        str,
        Union[
            List[str],
            PaddedNSortedEpisodeBatch,
            List[List[int]],
            List[List[float]],
        ],
    ]:
        user_id, user_history, item_index_episode, reward_episode, _return = tuple(
            np.array(batch, dtype=object).T
        )

        padded_user_history, lengths = self.pad_sequence(user_history)
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        return {
            "user_id": list(user_id[sorted_idx]),
            "user_history": PaddedNSortedEpisodeBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            "item_index_episode": list(item_index_episode[sorted_idx]),
            "reward_episode": list(reward_episode[sorted_idx]),
            "return": torch.from_numpy(_return[sorted_idx].astype(np.float32)).view(
                len(batch), -1
            ),
        }

    def to(self, batch: Dict, device: torch.device) -> Dict:
        if self.train:
            return {k: v.to(device) for k, v in batch.items()}
        else:
            non_tensors = ("user_id", "item_index_episode", "reward_episode")
            batch_on_device = {
                k: v.to(device) for k, v in batch.items() if k not in non_tensors
            }
            for k in non_tensors:
                batch_on_device[k] = batch[k]
            return batch_on_device


class RetailrocketEpisodeLoader(DataLoader):
    def __init__(
        self,
        dataset: RetailrocketEpisodeDataset,
        train: bool,
        discount_factor: float = 0.01,
        *args,
        **kargs
    ):
        self.dataset = dataset
        self.n_items = dataset.item_index_map.shape[0]
        self.train = train
        self.gamma = 1.0 - discount_factor
        if self.train is True:
            super().__init__(
                dataset=self.dataset, collate_fn=self.train_collate_func, *args, **kargs
            )
        else:
            super().__init__(
                dataset=self.dataset, collate_fn=self.eval_collate_func, *args, **kargs
            )

    def _backpad_sequence(
        self, sequences: Sequence[torch.FloatTensor]
    ) -> Tuple[torch.FloatTensor, torch.LongTensor]:
        lengths = torch.LongTensor([seq.size(0) for seq in sequences])
        max_length = lengths.max()
        padded = torch.stack(
            [
                torch.cat(
                    [seq, torch.zeros(max_length - seq.size(0), seq.size(1))]
                ).float()
                for seq in sequences
            ]
        )
        return padded, lengths

    def _n_hot_encode(
        self, item_seq: torch.LongTensor, reward_seq: torch.FloatTensor
    ) -> torch.FloatTensor:
        encoded = one_hot(item_seq, num_classes=self.n_items)
        encoded = encoded * reward_seq.unsqueeze(1).expand(-1, encoded.size(1))
        return encoded

    def _compute_return(self, rewards: Sequence[float]) -> float:
        gammas = self.gamma ** np.arange(len(rewards))
        return float(gammas @ np.array(rewards))

    def _compute_return_at_t(self, reward_episode: Sequence[float]) -> List[float]:
        return [
            self._compute_return(reward_episode[i:]) for i in range(len(reward_episode))
        ]

    def train_collate_func(
        self, batch: List[np.ndarray]
    ) -> Dict[str, Union[PackedSequence, np.ndarray]]:
        _, item_episodes, reward_episodes = tuple(np.array(batch, dtype=object).T)

        histories_encoded = [
            self._n_hot_encode(
                torch.LongTensor(item_ep[:-1]), torch.FloatTensor(reward_ep[:-1])
            )
            for item_ep, reward_ep in zip(item_episodes, reward_episodes)
        ]

        padded_histories, lengths = self._backpad_sequence(histories_encoded)
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        return_at_t = np.array(
            [self._compute_return_at_t(reward_ep) for reward_ep in reward_episodes],
            dtype=object,
        )

        return {
            "pack_padded_histories": pack_padded_sequence(
                input=padded_histories[sorted_idx],
                lengths=sorted_lengths,
                batch_first=True,
            ),
            "item_episodes": item_episodes[sorted_idx],
            "return_at_t": return_at_t[sorted_idx],
        }

    def eval_collate_func(
        self, batch: List[np.ndarray]
    ) -> Dict[str, Union[PackedSequence, np.ndarray, List[int]]]:
        data = self.train_collate_func(batch)
        data["user_id"] = list(tuple(np.array(batch, dtype=object).T)[0])
        return data

    def to(self, batch: Dict, device: torch.device) -> Dict:
        if self.train is True:
            non_tensors = ("item_episodes", "return_at_t")
        else:
            non_tensors = ("item_episodes", "return_at_t", "user_id")

        on_device = {k: v.to(device) for k, v in batch.items() if k not in non_tensors}

        for k in non_tensors:
            on_device[k] = batch[k]

        return on_device
