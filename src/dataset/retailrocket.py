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

    train_cols = ["user_history", "return", "item_index"]
    eval_cols = ["visitorid", "user_history", "item_index_episode", "reward_episode"]

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
        category_id: str = None,
    ):
        self.data_path = data_path

        self.train = train
        self.split_ratio = split_ratio

        self.cutoffs = sequence_length_cutoffs
        self.n_samples = n_samples
        self.sample_seed = randint(0, 9)
        np.random.seed(self.sample_seed)

        self.category_id = category_id

        self.discount_factor = discount_factor

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
                train_item_index_map
            ), "Item-index mapping from train sequence should be provided for evaluation dataset."
            self.item_index_map = train_item_index_map
        self.user_action_index_map = self._build_user_action_index_map(
            self.item_index_map
        )

        # 3. Build final data
        self.data = self._build_episodic_data(spark=spark)

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
            preprocessed = preprocessed.iloc[: ceil(n_records * self.split_ratio)]
        else:
            preprocessed = preprocessed.iloc[
                ceil(n_records * (1 - self.split_ratio)) :  # NoQA
            ]

        # 3. Alleviate user's history length imbalance (only when training)
        if self.train:
            user_history_len = preprocessed.groupby("visitorid")[["timestamp"]].count()
            user_history_len.rename(columns={"timestamp": "n_history"}, inplace=True)

            users_w_short_seq = user_history_len[
                (self.cutoffs[0][0] <= user_history_len.n_history)
                & (user_history_len.n_history <= self.cutoffs[0][1])
            ].index.to_numpy()
            users_w_long_seq = user_history_len[
                (self.cutoffs[1][0] <= user_history_len.n_history)
                & (user_history_len.n_history <= self.cutoffs[1][1])
            ].index.to_numpy()
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
            preprocessed = preprocessed[preprocessed.visitorid.isin(users_sampled)]

        # 4. Assign reward values
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
        df = self.logs[["itemid"]]
        df.drop_duplicates(inplace=True)
        df.sort_values(by="itemid", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.rename_axis("item_index", inplace=True)
        df.reset_index(drop=False, inplace=True)

        return df

    @staticmethod
    @udf(FloatType())
    def _compute_return(rewards: List[float], discount_factor: float) -> float:
        gammas = (1.0 - discount_factor) ** np.arange(len(rewards))
        return float(gammas @ np.array(rewards))

    def _build_episodic_data(self, spark: SparkSession) -> pd.DataFrame:
        index_merged = self.logs.merge(
            self.item_index_map, on="itemid", how="inner"
        ).merge(self.user_action_index_map, on=["itemid", "event"], how="inner")

        spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        sdf = spark.createDataFrame(index_merged)

        with_user_history = sdf.withColumn(
            "user_history",
            collect_list("user_action_index").over(
                W.partitionBy("visitorid")
                .orderBy("time")
                .rowsBetween(W.unboundedPreceding, -1)
            ),
        ).filter(size("user_history") > 0)

        if self.train is True:
            episodes_df = (
                with_user_history.withColumn(
                    "reward_episode",
                    collect_list("reward").over(
                        W.partitionBy("visitorid")
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
                        W.partitionBy("visitorid")
                        .orderBy("time")
                        .rowsBetween(W.currentRow, W.unboundedFollowing)
                    ),
                )
                .withColumn(
                    "reward_episode",
                    collect_list("reward").over(
                        W.partitionBy("visitorid")
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


class RetailrocketDataLoader(DataLoader):
    padding_signal = -1

    def __init__(self, train: bool, dataset: Dataset, *args, **kargs):
        self.train = train
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
        user_history, _return, item_index = tuple(np.array(batch, dtype=object).T)

        padded_user_history, lengths = self.pad_sequence(user_history)
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        return {
            "user_history": PaddedNSortedUserHistoryBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            "return": torch.from_numpy(_return.astype(np.float32)).view(batch_size, -1),
            "item_index": torch.from_numpy(item_index.astype(np.int64)).view(
                batch_size, -1
            ),
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
        user_id, user_history, item_index_episode, reward_episode = tuple(
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
        }

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
