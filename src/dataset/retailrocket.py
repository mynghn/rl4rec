import os
from math import ceil
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .custom_typings import PaddedNSortedUserHistoryBatch


class RetailrocketDataset(Dataset):
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
        discount_factor: float,
        days_in_episode: int,
        category_id: str = None,
        train_item_index_map: pd.DataFrame = None,
    ):
        self.data_path = data_path

        self.train = train
        self.split_ratio = split_ratio

        self.category_id = category_id

        self.discount_factor = discount_factor
        self.days_in_episode = days_in_episode

        # 0. Fetch Raw data
        self.events_df = self._get_events_df()
        self.item_category_df = self._get_item_category_df()

        # 1. Build event logs
        self.logs = self._build_event_logs()

        # 2. Build index maps
        if self.train is True:
            self.item_index_map = self._build_item_index_map()
        else:
            assert (
                train_item_index_map is not None
            ), "Item-index mapping from train sequence should be provided for evaluation dataset."
            self.item_index_map = train_item_index_map
        self.user_action_index_map = self._build_user_action_index_map(
            self.item_index_map
        )

        # 3. Build final data
        self.max_unixtime = self.logs.timestamp.max()
        (
            self.df,
            self.lifetime_timestamps_book,
            self.lifetime_items_book,
            self.lifetime_user_actions_book,
            self.lifetime_rewards_book,
        ) = self._build_data()

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

        # 3. Assign reward values
        preprocessed["reward"] = preprocessed.event.map(self.reward_map)

        return preprocessed

    def _build_user_action_index_map(
        self, item_index_map: pd.DataFrame
    ) -> pd.DataFrame:
        user_action_index_map = item_index_map.copy()
        user_action_index_map.drop("item_index", axis=1, inplace=True)
        user_action_index_map["event"] = user_action_index_map.itemid.map(
            lambda _: list(self.reward_map.keys())
        )
        user_action_index_map = user_action_index_map.explode(column="event")
        user_action_index_map.sort_values(by=["itemid", "event"], inplace=True)
        user_action_index_map.reset_index(drop=True, inplace=True)
        user_action_index_map.rename_axis("user_action_index", inplace=True)
        user_action_index_map.reset_index(drop=False, inplace=True)

        return user_action_index_map

    def _build_item_index_map(self) -> pd.DataFrame:
        df = self.logs[["itemid"]].copy()
        df.drop_duplicates(inplace=True)
        df.sort_values(by="itemid", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename_axis("item_index", inplace=True)
        df.reset_index(drop=False, inplace=True)

        return df

    def _build_data(
        self,
    ) -> Tuple[
        pd.DataFrame,
        Dict[str, List[int]],
        Dict[str, List[int]],
        Dict[str, List[int]],
        Dict[str, List[float]],
    ]:
        # 1. Merge indices
        df = self.logs.merge(self.item_index_map, on="itemid", how="inner").merge(
            self.user_action_index_map, on=["itemid", "event"], how="inner"
        )

        # 2. Assign event index within user history
        df.sort_values(by=["visitorid", "timestamp"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        groupedby_user = self.df.groupby(by="visitorid")
        df["event_index_in_user_history"] = groupedby_user.cumcount()

        # 3. Build lifetime sequence books by users
        lifetime_timestamps_book = groupedby_user["timestamp"].apply(list).to_dict()
        lifetime_items_book = groupedby_user["item_index"].apply(list).to_dict()
        lifetime_user_actions_book = (
            groupedby_user["user_action_index"].apply(list).to_dict()
        )
        lifetime_rewards_book = groupedby_user["reward"].apply(list).to_dict()

        # 3. Filter insufficient records
        unixtime_interval = self.days_in_episode * 86400 * 1000
        df = df[
            (df["event_index_in_user_history"] > 0)
            & (df["timestamp"] <= self.max_unixtime - unixtime_interval)
        ]

        df.reset_index(drop=True, inplace=True)

        return (
            df,
            lifetime_timestamps_book,
            lifetime_items_book,
            lifetime_user_actions_book,
            lifetime_rewards_book,
        )

    def __len__(self) -> int:
        return self.df.shape[0]

    def __getitem__(self, idx: int) -> List[Union[List[int], int, float]]:
        row = self.df.iloc[idx]
        user_id = row["visitorid"]
        index_in_history = row["event_index_in_user_history"]

        user_history = self.lifetime_user_actions_book[user_id][:index_in_history]

        timestamp_followings = self.lifetime_timestamps_book[user_id][index_in_history:]
        reward_followings = self.lifetime_rewards_book[user_id][index_in_history:]

        reward_episode = self._slice_episode(
            followings=reward_followings, timestamps=timestamp_followings
        )
        episodic_return = self._compute_return(rewards=reward_episode)

        if self.train is True:
            return [
                user_history,
                row["item_index"],
                episodic_return,
            ]
        else:
            item_followings = self.lifetime_items_book[user_id][index_in_history:]
            item_episode = self._slice_episode(
                followings=item_followings, timestamps=timestamp_followings
            )
            return [
                user_id,
                user_history,
                item_episode,
                reward_episode,
                episodic_return,
            ]

    def _slice_episode(self, followings: List[float], timestamps: List[int]) -> float:
        episode_length = self._get_episode_length(timestamps=timestamps)
        return followings[:episode_length]

    def _compute_return(self, rewards: List[float]) -> float:
        gammas = (1.0 - self.discount_factor) ** np.arange(len(rewards))
        return gammas @ np.array(rewards)

    def _get_episode_length(self, timestamps: List[int]) -> int:
        end_unixtime = timestamps[0] + (self.days_in_episode * 86400 * 1000)
        length = 1
        for ts in timestamps[1:]:
            if ts <= end_unixtime:
                length += 1
            else:
                break
        return length


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
            PaddedNSortedUserHistoryBatch,
            torch.LongTensor,
            torch.FloatTensor,
        ],
    ]:
        batch_size = len(batch)
        user_history, item_index, _return = tuple(np.array(batch, dtype=object).T)

        padded_user_history, lengths = self.pad_sequence(user_history)
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        return {
            "user_history": PaddedNSortedUserHistoryBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            "item_index": torch.from_numpy(item_index.astype(np.int64)).view(
                batch_size, -1
            ),
            "return": torch.from_numpy(_return.astype(np.float32)).view(batch_size, -1),
        }

    def eval_collate_func(
        self, batch: List[np.ndarray]
    ) -> Dict[
        str,
        Union[
            List[str],
            PaddedNSortedUserHistoryBatch,
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
            "user_id": list(user_id),
            "user_history": PaddedNSortedUserHistoryBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            "item_index_episode": list(item_index_episode),
            "reward_episode": list(reward_episode),
            "return": torch.from_numpy(_return.astype(np.float32)).view(len(batch), -1),
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
