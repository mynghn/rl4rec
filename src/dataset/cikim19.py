import os
from random import randint
from typing import Any, Dict, List, Sequence, Tuple, Union

import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    array,
    col,
    collect_list,
    count,
    explode,
    lit,
    monotonically_increasing_id,
    percent_rank,
    size,
    udf,
    when,
)
from pyspark.sql.types import (
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


class CIKIM19Dataset(Dataset):
    users_fetching_schema = StructType(
        [
            StructField("user", StringType()),
            StructField("sex", StringType()),
            StructField("age", IntegerType()),
            StructField("pur_power", IntegerType()),
        ]
    )
    items_fetching_schema = StructType(
        [
            StructField("item", StringType()),
            StructField("category", StringType()),
            StructField("shop", StringType()),
            StructField("brand", StringType()),
        ]
    )
    events_fetching_schema = StructType(
        [
            StructField("user", StringType()),
            StructField("item", StringType()),
            StructField("event", StringType()),
            StructField("time", LongType()),
        ]
    )

    reward_map = {"pv": 1.0, "fav": 2.0, "cart": 3.0, "buy": 5.0}

    user_feature_cols = ("sex", "age", "purpower")
    item_feature_cols = ("category", "brand", "shop")

    train_cols = (
        "user_history",
        "user_feature_index",
        "item_feature_index",
        "return",
        "item_index",
    )
    eval_cols = (
        "user",
        "user_history",
        "user_feature_index",
        "item_feature_index",
        "item_index_episode",
        "reward_episode",
    )

    def __init__(
        self,
        data_path: str,
        spark: SparkSession,
        train: bool,
        split_ratio: float,
        train_item_index_map: DataFrame = None,
        train_user_feature_index_map: DataFrame = None,
        train_item_feature_index_map: DataFrame = None,
        sequence_length_cutoffs: Tuple[Tuple[int, int], Tuple[int, int]] = (
            (6, 50),
            (51, 200),
        ),
        n_samples: Tuple(int, int) = (20000, 60000),
        category_id: str = None,
        discount_factor: float = 1e-2,
    ):
        self.data_path = data_path
        self.spark = spark

        self.train = train
        self.split_ratio = split_ratio

        self.cutoffs = sequence_length_cutoffs
        self.n_samples = n_samples
        self.sample_seed = randint(0, 9)

        self.category_id = category_id

        self.discount_factor = discount_factor

        self.logs: DataFrame = self._build_event_logs()

        if self.train is True:
            self.item_index_map = self._build_item_index_map()
            self.user_feature_index_map = self._build_user_feature_index_map()
            self.item_feature_index_map = self._build_item_feature_index_map()
        else:
            assert (
                train_item_index_map
                and train_user_feature_index_map
                and train_item_feature_index_map
            ), "Index mappings from train sequence should be provided for evaluation dataset."
            self.item_index_map = train_item_index_map
            self.user_feature_index_map = train_user_feature_index_map
            self.item_feature_index_map = train_item_feature_index_map
        self.user_action_index_map = self._build_user_action_index_map(
            self.item_index_map
        )

        self.data: np.ndarray = self._build_episodic_data()

    def _build_event_logs(self) -> DataFrame:
        users_df = self.spark.read.schema(self.users_fetching_schema).csv(
            os.path.join(self.data_path, "user.csv")
        )
        items_df = self.spark.read.schema(self.items_fetching_schema).csv(
            os.path.join(self.data_path, "item.csv")
        )
        events_df = self.spark.read.schema(self.events_fetching_schema).csv(
            os.path.join(self.data_path, "user_behavior.csv")
        )

        # 1. Split data
        if self.train is True:
            preprocessed = (
                events_df.withColumn(
                    "percent_rank", percent_rank().over(W.partitionBy().orderBy("time"))
                )
                .filter(col("percent_rank") <= self.split_ratio)
                .drop("percent_rank")
            )
        else:
            preprocessed = (
                events_df.withColumn(
                    "percent_rank", percent_rank().over(W.partitionBy().orderBy("time"))
                )
                .filter(col("percent_rank") > 1.0 - self.split_ratio)
                .drop("percent_rank")
            )

        # 2. Alleviate user's sequence length imbalance
        user_seq_len = preprocessed.groupBy("user").agg(
            count("*").alias("sequence_length")
        )
        users_w_short_seq = user_seq_len.filter(
            col("sequence_length").between(*self.cutoffs[0])
        ).drop("sequence_length")
        users_w_long_seq = user_seq_len.filter(
            col("sequence_length").between(*self.cutoffs[-1])
        ).drop("sequence_length")
        n_short = users_w_short_seq.count()
        n_long = users_w_long_seq.count()

        if n_short < self.n_samples[0] and n_long < self.n_samples[-1]:
            short_long_ratio = self.n_samples[0] / self.n_samples[-1]
            if n_short / n_long - 1e-2 < short_long_ratio < n_short / n_long + 1e-2:
                users_w_short_seq_sampled = users_w_short_seq
                users_w_long_seq_sampled = users_w_long_seq
            elif short_long_ratio > n_short / n_long:
                users_w_short_seq_sampled = users_w_short_seq
                _ratio_long = n_short / short_long_ratio / n_long
                users_w_long_seq_sampled = users_w_long_seq.sample(
                    _ratio_long, self.sample_seed
                )
            elif short_long_ratio < n_short / n_long:
                _ratio_short = n_long * short_long_ratio / n_short
                users_w_short_seq_sampled = users_w_short_seq.sample(
                    _ratio_short, self.sample_seed
                )
                users_w_long_seq_sampled = users_w_long_seq
        elif n_short < self.n_samples[0]:
            users_w_short_seq_sampled = users_w_short_seq
            _ratio_long = n_short * self.n_samples[-1] / self.n_samples[0] / n_long
            users_w_long_seq_sampled = users_w_long_seq.sample(
                _ratio_long, self.sample_seed
            )
        elif n_long < self.n_samples[-1]:
            _ratio_short = n_long * self.n_samples[0] / self.n_samples[-1] / n_short
            users_w_short_seq_sampled = users_w_short_seq.sample(
                _ratio_short, self.sample_seed
            )
            users_w_long_seq_sampled = users_w_long_seq
        else:
            _ratio_short = self.n_samples[0] / n_short
            users_w_short_seq_sampled = users_w_short_seq.sample(
                _ratio_short, self.sample_seed
            )
            _ratio_long = self.n_samples[-1] / n_long
            users_w_long_seq_sampled = users_w_long_seq.sample(
                _ratio_long, self.sample_seed
            )

        users_sampled = users_w_short_seq_sampled.union(users_w_long_seq_sampled)
        preprocessed = preprocessed.join(users_sampled, ["user"], "inner")

        # 3. Assign reward values
        preprocessed = preprocessed.withColumn(
            "reward",
            when(col("event") == "pv", lit(1.0))
            .when(col("event") == "fav", lit(2.0))
            .when(col("event") == "cart", lit(3.0))
            .when(col("event") == "buy", lit(5.0))
            .otherwise(lit(None)),
        ).filter(col("reward").isNotNull())

        # 4. Merge & Collect only events with items in specific category
        if self.category_id:
            preprocessed = (
                preprocessed.join(users_df, on=["user"], how="left")
                .join(
                    items_df.withColumn(
                        "in_category", col("category") == self.category_id
                    ),
                    on=["item"],
                    how="left",
                )
                .filter("in_category")
            )
        else:
            preprocessed = preprocessed.join(users_df, on=["user"], how="left").join(
                items_df,
                on=["item"],
                how="left",
            )

        # 5. Age binning
        preprocessed = preprocessed.withColumn(
            "age",
            when(col("age") < 30, lit(0))
            .when(col("age") < 50, lit(1))
            .when(col("age") < 70, lit(2))
            .otherwise(lit(3)),
        )

        return preprocessed.select(
            "time",
            "user",
            "sex",
            "age",
            "pur_power",
            "item",
            "category",
            "shop",
            "brand",
            "event",
            "reward",
        )

    def _build_user_action_index_map(self, item_index_map: DataFrame) -> DataFrame:
        return (
            item_index_map.select("item")
            .withColumn(
                "event", explode(array([lit(act) for act in self.reward_map.keys()]))
            )
            .coalesce(1)
            .orderBy("item", "event")
            .withColumn("user_action_index", monotonically_increasing_id())
        )

    def _build_user_feature_index_map(self) -> DataFrame:
        return (
            self.logs.select(*self.user_feature_cols)
            .distinct()
            .coalesce(1)
            .orderBy(*self.user_feature_cols)
            .withColumn("user_feature_index", monotonically_increasing_id())
        )

    def _build_item_feature_index_map(self) -> DataFrame:

        return (
            self.logs.select(*self.item_feature_cols)
            .distinct()
            .coalesce(1)
            .orderBy(*self.item_feature_cols)
            .withColumn("item_feature_index", monotonically_increasing_id())
        )

    def _build_item_index_map(self) -> DataFrame:
        return (
            self.logs.select("item")
            .distinct()
            .coalesce(1)
            .orderBy("item")
            .withColumn("item_index", monotonically_increasing_id())
        )

    @staticmethod
    @udf(FloatType())
    def _compute_return(rewards: List[float], discount_factor: float) -> float:
        gammas = (1.0 - discount_factor) ** np.arange(len(rewards))
        return float(gammas @ np.array(rewards))

    def _build_episodic_data(self) -> np.ndarray:
        logs_template = (
            self.logs.join(self.item_index_map, ["item"])
            .join(self.user_action_index_map, ["item", "event"])
            .join(self.user_feature_index_map, [*self.user_feature_cols])
            .join(self.item_feature_index_map, [*self.item_feature_cols])
            .withColumn(
                "user_history",
                collect_list("user_action_index").over(
                    W.partitionBy("user")
                    .orderBy("time")
                    .rowsBetween(W.unboundedPreceding, -1)
                ),
            )
            .filter(size("user_history") > 0)
        )

        if self.train is True:
            episodes_df = (
                logs_template.withColumn(
                    "reward_episode",
                    collect_list("reward").over(
                        W.partitionBy("user")
                        .orderBy("time")
                        .rowsBetween(W.currentRow, W.unboundedFollowing)
                    ),
                )
                .withColumn("discount_factor", lit(self.discount_factor))
                .withColumn(
                    "return", self._compute_return("reward_episode", "discount_factor")
                )
                .select(*self.train_cols)
            )
            return np.array(
                [(row[col] for col in self.train_cols) for row in episodes_df.collect()]
            )
        else:
            episodes_df = (
                logs_template.withColumn(
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
            return np.array(
                [(row[col] for col in self.eval_cols) for row in episodes_df.collect()]
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Any:
        return self.data[idx]


class CIKIM19DataLoader(DataLoader):
    padding_signal = -1

    def __init__(self, train: bool, *args, **kargs):
        self.train = train
        if self.train is True:
            super().__init__(*args, **kargs, collate_fn=self.train_collate_func)
        else:
            super().__init__(*args, **kargs, collate_fn=self.eval_collate_func)

    @staticmethod
    def pad_sequence(
        user_history: Sequence[Sequence[int]],
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

    @staticmethod
    def train_collate_func(
        batch: List[Tuple[List[int], int, int, float, int]],
    ) -> Dict[
        str,
        Union[
            PaddedNSortedUserHistoryBatch,
            torch.LongTensor,
            torch.FloatTensor,
        ],
    ]:
        batch_size = len(batch)
        (
            user_history,
            user_feature_index,
            item_feature_index,
            _return,
            item_index,
        ) = tuple(np.array(batch, dtype=object).T)

        padded_user_history, lengths = CIKIM19DataLoader.pad_sequence(user_history)
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        return {
            "user_history": PaddedNSortedUserHistoryBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            "user_feature_index": torch.from_numpy(
                user_feature_index.astype(np.int64)
            ).view(batch_size, -1),
            "item_feature_index": torch.from_numpy(
                item_feature_index.astype(np.int64)
            ).view(batch_size, -1),
            "return": torch.from_numpy(_return.astype(np.float32)).view(batch_size, -1),
            "item_index": torch.from_numpy(item_index.astype(np.int64)).view(
                batch_size, -1
            ),
        }

    @staticmethod
    def eval_collate_func(
        batch: List[Tuple[str, List[int], int, int, List[str], List[float]]],
    ) -> Dict[
        str,
        Union[
            List[str],
            PaddedNSortedUserHistoryBatch,
            torch.LongTensor,
            List[List[str]],
            List[torch.FloatTensor],
        ],
    ]:
        batch_size = len(batch)
        (
            user_id,
            user_history,
            user_feature_index,
            item_feature_index,
            item_index_episode,
            reward_episode,
        ) = tuple(np.array(batch, dtype=object).T)

        padded_user_history, lengths = CIKIM19DataLoader.pad_sequence(user_history)
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        return {
            "user_id": list(user_id),
            "user_history": PaddedNSortedUserHistoryBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            "user_feature_index": torch.from_numpy(
                user_feature_index.astype(np.int64)
            ).view(batch_size, -1),
            "item_feature_index": torch.from_numpy(
                item_feature_index.astype(np.int64)
            ).view(batch_size, -1),
            "item_index_episode": [list(seq) for seq in item_index_episode],
            "reward_episode": [torch.FloatTensor(seq) for seq in reward_episode],
        }
