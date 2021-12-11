import datetime
import os
from math import ceil
from random import randint
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, collect_list, from_unixtime, lit, size, udf
from pyspark.sql.types import (
    ArrayType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
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

    spark_schema = StructType(
        [
            StructField("timestamp", LongType()),
            StructField("visitorid", StringType()),
            StructField("event", StringType()),
            StructField("itemid", StringType()),
            StructField("categoryid", StringType()),
            StructField("reward", FloatType()),
            StructField("item_index", IntegerType()),
            StructField("user_action_index", IntegerType()),
        ]
    )

    train_cols = ["user_history", "return", "item_index"]
    eval_cols = ["visitorid", "user_history", "item_index_episode", "reward_episode"]

    def __init__(
        self,
        spark: SparkSession,
        data_path: str,
        train: bool,
        split_ratio: float,
        category_id: str = None,
        train_item_index_map: pd.DataFrame = None,
        sequence_length_cutoffs: Tuple[Tuple[int, int], Tuple[int, int]] = (
            (6, 50),
            (51, 200),
        ),
        n_samples: Tuple[int, int] = (20000, 60000),
        discount_factor: float = 1e-2,
        episode_length: int = 7,
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
        self.episode_length = episode_length

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
            preprocessed = preprocessed.iloc[ceil(n_records * (1 - self.split_ratio)) :]

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

    @staticmethod
    @udf(FloatType())
    def _compute_return(rewards: List[float], discount_factor: float) -> float:
        gammas = (1.0 - discount_factor) ** np.arange(len(rewards))
        return float(gammas @ np.array(rewards))

    @staticmethod
    @udf(
        ArrayType(
            StructType(
                [
                    StructField("item_index", IntegerType()),
                    StructField("reward", FloatType()),
                ]
            )
        )
    )
    def _build_episode(
        timestamps: List[datetime.datetime],
        items: List[int],
        rewards: List[float],
        length: int,
        max_time: datetime.datetime,
    ) -> Optional[List[Dict[str, Union[int, float]]]]:
        start_time = timestamps.pop(0)
        end_time = start_time + datetime.timedelta(days=length)
        if end_time <= max_time:
            sliced = [{"item_index": items.pop(0), "reward": rewards.pop(0)}]
            for ts, item_index, reward in zip(timestamps, items, rewards):
                if ts <= end_time:
                    sliced.append({"item_index": item_index, "reward": reward})
                else:
                    break
            return sliced

    def _build_episodic_data(self, spark: SparkSession) -> pd.DataFrame:
        index_merged = self.logs.merge(
            self.item_index_map, on="itemid", how="inner"
        ).merge(self.user_action_index_map, on=["itemid", "event"], how="inner")

        spark.conf.set("spark.sql.execution.arrow.enabled", "true")
        sdf = spark.createDataFrame(index_merged, schema=self.spark_schema)

        with_historyNepisode = (
            sdf.withColumn(
                "user_history",
                collect_list("user_action_index").over(
                    W.partitionBy("visitorid")
                    .orderBy("timestamp")
                    .rowsBetween(W.unboundedPreceding, -1)
                ),
            )
            .filter(size("user_history") > 0)
            .withColumn(
                "timestamp", from_unixtime(col("timestamp") / 1000).cast("timestamp")
            )
            .withColumn(
                "following_timestamps",
                collect_list("timestamp").over(
                    W.partitionBy("visitorid")
                    .orderBy("timestamp")
                    .rowsBetween(W.currentRow, W.unboundedFollowing)
                ),
            )
            .withColumn(
                "following_items",
                collect_list("item_index").over(
                    W.partitionBy("visitorid")
                    .orderBy("timestamp")
                    .rowsBetween(W.currentRow, W.unboundedFollowing)
                ),
            )
            .withColumn(
                "following_rewards",
                collect_list("reward").over(
                    W.partitionBy("visitorid")
                    .orderBy("timestamp")
                    .rowsBetween(W.currentRow, W.unboundedFollowing)
                ),
            )
            .withColumn(
                "episode",
                self._build_episode(
                    "following_timestamps",
                    "following_items",
                    "following_rewards",
                    lit(self.episode_length),
                    lit(
                        datetime.datetime.fromtimestamp(
                            self.logs.timestamp.max() / 1000
                        )
                    ),
                ),
            )
        )

        if self.train is True:
            episodes_df = (
                with_historyNepisode.withColumn(
                    "reward_episode", col("episode").getItem("reward")
                )
                .withColumn(
                    "return",
                    self._compute_return("reward_episode", lit(self.discount_factor)),
                )
                .select(*self.train_cols)
            )
        else:
            episodes_df = (
                with_historyNepisode.withColumn(
                    "item_index_episode", col("episode").getItem("item_index")
                )
                .withColumn("reward_episode", col("episode").getItem("reward"))
                .select(*self.eval_cols)
            )

        return episodes_df.toPandas()

    def __len__(self) -> int:
        return self.data.shape[0]

    def __getitem__(self, idx: Union[int, List[int]]) -> np.ndarray:
        return self.data.iloc[idx]


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
            non_tensors = ("user_id", "item_index_episode", "reward_episode")
            batch_on_device = {
                k: v.to(device) for k, v in batch.items() if k not in non_tensors
            }
            for k in non_tensors:
                batch_on_device[k] = batch[k]
            return batch_on_device
