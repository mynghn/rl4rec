import os
from math import ceil
from random import randint
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list, lit, size, udf
from pyspark.sql.types import FloatType
from pyspark.sql.window import Window as W
from torch.utils.data import DataLoader, Dataset

from .custom_typings import PaddedNSortedUserHistoryBatch


class CIKIM19Dataset(Dataset):
    users_fetching_dtypes = {"user": str, "sex": str, "age": int, "pur_power": int}
    items_fetching_dtypes = {"item": str, "category": str, "shop": str, "brand": str}
    events_fetching_dtypes = {"user": str, "item": str, "event": str, "time": int}

    reward_map = {"pv": 1.0, "fav": 2.0, "cart": 3.0, "buy": 5.0}

    user_feature_cols = ["sex", "age", "pur_power"]
    item_feature_cols = ["category", "brand", "shop"]

    train_cols = ["user_history", "return", "item_index"]
    eval_cols = ["user", "user_history", "item_index_episode", "reward_episode"]

    def __init__(
        self,
        spark: SparkSession,
        data_path: str,
        train: bool,
        split_ratio: float,
        train_item_index_map: pd.DataFrame = None,
        sequence_length_cutoffs: Tuple[Tuple[int, int], Tuple[int, int]] = (
            (6, 50),
            (51, 200),
        ),
        n_samples: Tuple[int, int] = (20000, 60000),
        discount_factor: float = 1e-2,
        events_truncated: bool = True,
        events_splitted: bool = False,
        events_suffix_list: List[str] = None,
        category_id: str = None,
        user_feature: bool = False,
        item_feature: bool = False,
        max_records: int = None,
    ):
        self.data_path = data_path

        self.train = train
        self.split_ratio = split_ratio

        self.cutoffs = sequence_length_cutoffs
        self.n_samples = n_samples
        self.sample_seed = randint(0, 9)

        self.category_id = category_id
        self.user_feature_enabled = user_feature
        self.item_feature_enabled = item_feature

        self.discount_factor = discount_factor

        self.max_records = max_records

        # 0. Fetch Raw data
        self.users_df: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_path, "user.csv"),
            header=None,
            names=self.users_fetching_dtypes.keys(),
            dtype=self.users_fetching_dtypes,
        )
        self.users_df["age"] = self.users_df["age"].apply(self.age_binning)

        self.items_df: pd.DataFrame = pd.read_csv(
            os.path.join(self.data_path, "item.csv"),
            header=None,
            names=self.items_fetching_dtypes.keys(),
            dtype=self.items_fetching_dtypes,
        )
        if events_truncated is True:
            self.events_df: pd.DataFrame = pd.read_csv(
                os.path.join(self.data_path, "user_behavior_truncated.csv"),
                header=None,
                names=self.events_fetching_dtypes.keys(),
                dtype=self.events_fetching_dtypes,
            )
        elif events_splitted is True:
            assert events_suffix_list, "File suffix list should be provided"
            df_list = []
            for suffix in events_suffix_list:
                df_list.append(
                    pd.read_csv(
                        os.path.join(self.data_path, f"user_behavior_{suffix}"),
                        header=None,
                        names=self.events_fetching_dtypes.keys(),
                        dtype=self.events_fetching_dtypes,
                    )
                )
            self.events_df: pd.DataFrame = pd.concat(df_list)
        else:
            self.events_df: pd.DataFrame = pd.read_csv(
                os.path.join(self.data_path, "user_behavior.csv"),
                header=None,
                names=self.events_fetching_dtypes.keys(),
                dtype=self.events_fetching_dtypes,
            )

        # 1. Build event logs
        self.logs = self._build_event_logs()

        # 2. Build index maps
        if self.train is True:
            self.item_index_map = self._build_item_index_map()
        else:
            assert (
                train_item_index_map
            ), "Item-index mapping from train sequence should be provided for evaluation dataset."
            self.item_index_map = train_item_index_map
        self.user_action_index_map = self._build_user_action_index_map(
            self.item_index_map
        )
        if self.user_feature_enabled is True:
            self.user_feature_index_map = self._build_user_feature_index_map()
            self.train_cols.append("user_feature_index")
            self.eval_cols.append("user_feature_index")
        if self.item_feature_enabled is True:
            self.item_feature_index_map = self._build_item_feature_index_map()
            self.train_cols.append("item_feature_index")
            self.eval_cols.append("item_feature_index")

        # 3. Build final data
        self.data = self._build_episodic_data(spark=spark)

    @staticmethod
    def age_binning(age):
        if age < 30:
            return 1
        elif age < 50:
            return 2
        elif age < 70:
            return 3
        else:
            return 4

    def _build_event_logs(self) -> pd.DataFrame:
        # 1. Merge & Collect only events with items in specific category
        preprocessed = self.events_df.merge(self.users_df, on="user", how="left").merge(
            self.items_df, on="item", how="left"
        )
        if self.category_id:
            preprocessed = preprocessed[preprocessed.category == self.category_id]

        # 2. Split data
        preprocessed = (
            preprocessed.drop_duplicates().sort_values(by="time").reset_index(drop=True)
        )
        if self.max_records:
            n_records = min(self.max_records, preprocessed.shape[0])
            preprocessed = preprocessed[preprocessed.index < n_records]
        else:
            n_records = preprocessed.shape[0]
        if self.train is True:
            preprocessed = preprocessed[
                preprocessed.index < ceil(n_records * self.split_ratio)
            ]
        else:
            preprocessed = preprocessed[
                preprocessed.index >= ceil(n_records * (1 - self.split_ratio))
            ]

        # 3. Alleviate user's history length imbalance (only when training)
        if self.train:
            user_history_len = (
                preprocessed.groupby("user")[["time"]]
                .count()
                .rename(columns={"time": "n_history"})
            )
            users_w_short_seq = np.array(
                user_history_len[
                    (self.cutoffs[0][0] <= user_history_len.n_history)
                    & (user_history_len.n_history <= self.cutoffs[0][1])
                ].index
            )
            users_w_long_seq = np.array(
                user_history_len[
                    (self.cutoffs[1][0] <= user_history_len.n_history)
                    & (user_history_len.n_history <= self.cutoffs[1][1])
                ].index
            )
            n_short = len(users_w_short_seq)
            n_long = len(users_w_long_seq)

            if n_short < self.n_samples[0] and n_long < self.n_samples[1]:
                short_long_ratio = self.n_samples[0] / self.n_samples[1]
                if n_short / n_long - 1e-2 < short_long_ratio < n_short / n_long + 1e-2:
                    users_w_short_seq_sampled = users_w_short_seq
                    users_w_long_seq_sampled = users_w_long_seq
                elif short_long_ratio > n_short / n_long:
                    users_w_short_seq_sampled = users_w_short_seq
                    _n_sample_long = ceil(n_short / short_long_ratio)
                    users_w_long_seq_sampled = np.random.choice(
                        users_w_long_seq, _n_sample_long, replace=False
                    )
                elif short_long_ratio < n_short / n_long:
                    _n_sample_short = ceil(n_long * short_long_ratio)
                    users_w_short_seq_sampled = np.random.choice(
                        users_w_short_seq, _n_sample_short, replace=False
                    )
                    users_w_long_seq_sampled = users_w_long_seq
            elif n_short < self.n_samples[0]:
                users_w_short_seq_sampled = users_w_short_seq
                _n_sample_long = ceil(n_short * self.n_samples[1] / self.n_samples[0])
                users_w_long_seq_sampled = np.random.choice(
                    users_w_long_seq, _n_sample_long, replace=False
                )
            elif n_long < self.n_samples[1]:
                _n_sample_short = ceil(n_long * self.n_samples[0] / self.n_samples[1])
                users_w_short_seq_sampled = np.random.choice(
                    users_w_short_seq, _n_sample_short, replace=False
                )
                users_w_long_seq_sampled = users_w_long_seq
            else:
                users_w_short_seq_sampled = np.random.choice(
                    users_w_short_seq, self.n_samples[0], replace=False
                )
                users_w_long_seq_sampled = np.random.choice(
                    users_w_long_seq, self.n_samples[1], replace=False
                )

            users_sampled = np.concatenate(
                [users_w_short_seq_sampled, users_w_long_seq_sampled]
            )
            preprocessed = preprocessed[preprocessed.user.isin(users_sampled)]

        # 4. Assign reward values
        preprocessed["reward"] = preprocessed.event.map(self.reward_map)

        return preprocessed

    def _build_user_action_index_map(
        self, item_index_map: pd.DataFrame
    ) -> pd.DataFrame:
        user_action_index_map = item_index_map.copy().drop("item_index", axis=1)
        user_action_index_map["event"] = user_action_index_map.item.map(
            lambda _: list(self.reward_map.keys())
        )
        user_action_index_map = user_action_index_map.explode(column="event")
        return (
            user_action_index_map.sort_values(by=["item", "event"])
            .reset_index(drop=True)
            .rename_axis("user_action_index")
            .reset_index(drop=False)
        )

    def _build_user_feature_index_map(self) -> pd.DataFrame:
        return (
            self.users_df[self.user_feature_cols]
            .drop_duplicates(subset=self.user_feature_cols)
            .sort_values(by=self.user_feature_cols)
            .reset_index(drop=True)
            .rename_axis("user_feature_index")
            .reset_index(drop=False)
        )

    def _build_item_feature_index_map(self) -> pd.DataFrame:

        return (
            self.items_df[self.item_feature_cols]
            .drop_duplicates(subset=self.item_feature_cols)
            .sort_values(by=self.item_feature_cols)
            .reset_index(drop=True)
            .rename_axis("item_feature_index")
            .reset_index(drop=False)
        )

    def _build_item_index_map(self) -> pd.DataFrame:
        return (
            self.logs[["item"]]
            .drop_duplicates()
            .sort_values(by="item")
            .reset_index(drop=True)
            .rename_axis("item_index")
            .reset_index(drop=False)
        )

    @staticmethod
    @udf(FloatType())
    def _compute_return(rewards: List[float], discount_factor: float) -> float:
        gammas = (1.0 - discount_factor) ** np.arange(len(rewards))
        return float(gammas @ np.array(rewards))

    def _build_episodic_data(self, spark: SparkSession) -> pd.DataFrame:
        index_merged = self.logs.merge(
            self.item_index_map, on="item", how="inner"
        ).merge(self.user_action_index_map, on=["item", "event"], how="inner")
        if self.user_feature_enabled:
            index_merged = index_merged.merge(
                self.user_feature_index_map, on=self.user_feature_cols, how="inner"
            )
        if self.item_feature_enabled:
            index_merged = index_merged.merge(
                self.item_feature_index_map, on=self.item_feature_cols, how="inner"
            )

        spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        sdf = spark.createDataFrame(index_merged)

        with_user_history = sdf.withColumn(
            "user_history",
            collect_list("user_action_index").over(
                W.partitionBy("user")
                .orderBy("time")
                .rowsBetween(W.unboundedPreceding, -1)
            ),
        ).filter(size("user_history") > 0)

        if self.train is True:
            episodes_df = (
                with_user_history.withColumn(
                    "reward_episode",
                    collect_list("reward").over(
                        W.partitionBy("user")
                        .orderBy("time")
                        .rowsBetween(W.currentRow, W.unboundedFollowing)
                    ),
                )
                .withColumn(
                    "return",
                    self._compute_return("reward_episode", lit(self.discount_factor)),
                )
                .select(*self.train_cols)
            )
        else:
            episodes_df = (
                with_user_history.withColumn(
                    "item_index_episode",
                    collect_list("item_index").over(
                        W.partitionBy("user")
                        .orderBy("time")
                        .rowsBetween(W.currentRow, W.unboundedFollowing)
                    ),
                )
                .withColumn(
                    "reward_episode",
                    collect_list("reward").over(
                        W.partitionBy("user")
                        .orderBy("time")
                        .rowsBetween(W.currentRow, W.unboundedFollowing)
                    ),
                )
                .select(*self.eval_cols)
            )

        return episodes_df.toPandas()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: Union[int, List[int]]) -> np.ndarray:
        return self.data.iloc[idx].to_numpy()


class CIKIM19DataLoader(DataLoader):
    padding_signal = -1

    def __init__(self, train: bool, dataset: Dataset, *args, **kargs):
        self.train = train
        self.user_feature_enabled = dataset.user_feature_enabled
        self.item_feature_enabled = dataset.item_feature_enabled
        if self.train is True:
            super().__init__(
                dataset=dataset,
                collate_fn=self.train_collate_func,
                *args,
                **kargs,
            )
        else:
            super().__init__(
                dataset=dataset,
                collate_fn=self.eval_collate_func,
                *args,
                **kargs,
            )

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
                        torch.zeros(max_length - len(item_seq))
                        + CIKIM19DataLoader.padding_signal,
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
        if self.user_feature_enabled and self.item_feature_enabled:
            (
                user_history,
                _return,
                item_index,
                user_feature_index,
                item_feature_index,
            ) = tuple(np.array(batch, dtype=object).T)
        elif self.user_feature_enabled:
            user_history, _return, item_index, user_feature_index = tuple(
                np.array(batch, dtype=object).T
            )
        elif self.item_feature_enabled:
            user_history, _return, item_index, item_feature_index = tuple(
                np.array(batch, dtype=object).T
            )
        else:
            user_history, _return, item_index = tuple(np.array(batch, dtype=object).T)

        padded_user_history, lengths = self.pad_sequence(user_history)
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        batch_dict = {
            "user_history": PaddedNSortedUserHistoryBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            "return": torch.from_numpy(_return.astype(np.float32)).view(batch_size, -1),
            "item_index": torch.from_numpy(item_index.astype(np.int64)).view(
                batch_size, -1
            ),
        }

        if self.user_feature_enabled:
            batch_dict["user_feature_index"] = torch.from_numpy(
                user_feature_index.astype(np.int64)
            ).view(batch_size, -1)
        if self.item_feature_enabled:
            batch_dict["item_feature_index"] = torch.from_numpy(
                item_feature_index.astype(np.int64)
            ).view(batch_size, -1)

        return batch_dict

    def eval_collate_func(
        self, batch: List[np.ndarray]
    ) -> Dict[
        str,
        Union[
            List[str],
            PaddedNSortedUserHistoryBatch,
            torch.LongTensor,
            List[List[int]],
            List[List[float]],
        ],
    ]:
        batch_size = len(batch)
        if self.user_feature_enabled and self.item_feature_enabled:
            (
                user_id,
                user_history,
                item_index_episode,
                reward_episode,
                user_feature_index,
                item_feature_index,
            ) = tuple(np.array(batch, dtype=object).T)
        elif self.user_feature_enabled:
            (
                user_id,
                user_history,
                item_index_episode,
                reward_episode,
                user_feature_index,
            ) = tuple(np.array(batch, dtype=object).T)
        elif self.item_feature_enabled:
            (
                user_id,
                user_history,
                item_index_episode,
                reward_episode,
                item_feature_index,
            ) = tuple(np.array(batch, dtype=object).T)
        else:
            user_id, user_history, item_index_episode, reward_episode = tuple(
                np.array(batch, dtype=object).T
            )

        padded_user_history, lengths = CIKIM19DataLoader.pad_sequence(user_history)
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        batch_dict = {
            "user_id": list(user_id),
            "user_history": PaddedNSortedUserHistoryBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            "item_index_episode": list(item_index_episode),
            "reward_episode": list(reward_episode),
        }

        if self.user_feature_enabled:
            batch_dict["user_feature_index"] = torch.from_numpy(
                user_feature_index.astype(np.int64)
            ).view(batch_size, -1)
        if self.item_feature_enabled:
            batch_dict["item_feature_index"] = torch.from_numpy(
                item_feature_index.astype(np.int64)
            ).view(batch_size, -1)

        return batch_dict

    def to(self, batch: Dict, device: torch.device) -> Dict:
        if self.train:
            return {k: v.to(device) for k, v in batch.items()}
        else:
            batch_on_device = {
                k: v.to(device)
                for k, v in batch.items()
                if k not in ("user_id", "item_index_episode", "reward_episode")
            }
            batch_on_device["user_id"] = batch["user_id"]
            batch_on_device["item_index_episode"] = [
                seq.to(device) for seq in batch["item_index_episode"]
            ]
            batch_on_device["reward_episode"] = [
                seq.to(device) for seq in batch["reward_episode"]
            ]
            return batch_on_device
