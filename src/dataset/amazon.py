import datetime
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    col,
    collect_list,
    countDistinct,
    from_unixtime,
    lit,
    monotonically_increasing_id,
    regexp_extract,
    regexp_replace,
    row_number,
    size,
    udf,
)
from pyspark.sql.types import (
    ArrayType,
    BooleanType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from pyspark.sql.window import Window as W
from torch.utils.data import DataLoader, Dataset


class AmazonReviewTrainDataset(Dataset):
    ratings_fetching_schema = StructType(
        [
            StructField("reviewerID", StringType()),
            StructField("asin", StringType()),
            StructField("overall", FloatType()),
            StructField("unixReviewTime", LongType()),
            StructField("verified", BooleanType()),
        ]
    )

    metadata_fetching_schema = StructType(
        [
            StructField("asin", StringType()),
            StructField("title", StringType()),
            StructField("brand", StringType()),
            StructField("category", ArrayType(StringType())),
            StructField("rank", StringType()),
            StructField("also_buy", ArrayType(StringType())),
            StructField("also_view", ArrayType(StringType())),
        ]
    )

    def __init__(
        self,
        data_path: str,
        category_name: str,
        spark: SparkSession,
        discount_factor: float,
        start_date: datetime.date = None,
        end_date: datetime.date = None,
        min_sequence_length: int = None,
        max_sequence_length: int = None,
    ):
        self.spark = spark
        self.category_name = category_name
        self.start_date = start_date
        self.end_date = end_date
        self.discount_factor = discount_factor
        self.threshold = (min_sequence_length, max_sequence_length)

        self.ratings: DataFrame = self._get_ratings_df(
            os.path.join(data_path, f"{self.category_name.replace(' ', '_')}.json")
        )

        self.item_index_map: DataFrame = self._build_item_index_map()
        self.data: np.ndarray = self._build_episodic_data()

    def _get_ratings_df(self, json_path: str) -> DataFrame:
        raw = self.spark.read.schema(self.ratings_fetching_schema).json(json_path)

        # 1. Filter unverified ratings
        preprocessed = raw.filter("verified").drop("verified")

        # 2. Collect only ratings within training time window
        if self.start_date:
            preprocessed = preprocessed.filter(
                from_unixtime("unixReviewTime").cast("date") >= self.start_date
            )
        if self.end_date:
            preprocessed = preprocessed.filter(
                from_unixtime("unixReviewTime").cast("date") <= self.end_date
            )

        # 3. Give up on ratings gave by same user to same item at same date and has different overall
        giveups_filtered = (
            preprocessed.groupBy("reviewerID", "asin", "unixReviewTime")
            .agg(countDistinct("overall").alias("num_overalls"))
            .filter(col("num_overalls") == 1)
            .drop("num_overalls")
        )
        preprocessed = preprocessed.join(
            giveups_filtered, ["reviewerID", "asin", "unixReviewTime"]
        )

        # 4. Preserve only the latest rating that user gave to same item
        preprocessed = (
            preprocessed.withColumn(
                "is_latest",
                row_number().over(
                    W.partitionBy("reviewerID", "asin").orderBy(
                        col("unixReviewTime").desc()
                    )
                )
                == lit(1),
            )
            .filter("is_latest")
            .drop("is_latest")
        )

        return preprocessed

    def _get_metadata_df(self, json_path: str) -> DataFrame:
        raw = self.spark.read.schema(self.metadata_fetching_schema).json(json_path)

        preprocessed = raw.withColumn(
            "rank",
            regexp_extract(regexp_replace("rank", r"[,]", ""), r"[\d]+", 0).cast("int"),
        )
        if self.category_name == "Movies and TV":
            preprocessed = preprocessed.withColumn(
                "brand", self._filter_movies_and_tv_brand_column("brand")
            )

        return preprocessed

    @staticmethod
    @udf(StringType())
    def _filter_movies_and_tv_brand_column(brand: str) -> Optional[str]:
        lowered = brand.lower()
        no_info = (
            "various",
            "n/a",
            ".",
            "\n",
            "none",
            "-",
            "*",
            "na",
            "artist not provided",
            "various artists",
            "learn more",
        )
        if lowered and lowered not in no_info:
            return lowered

    def _build_item_index_map(self) -> DataFrame:
        return (
            self.ratings.select("asin")
            .distinct()
            .coalesce(1)
            .orderBy("asin")
            .withColumn("item_index", monotonically_increasing_id())
        )

    @staticmethod
    @udf(FloatType())
    def _compute_return(rewards: List[float], discount_factor: float) -> float:
        gammas = (1.0 - discount_factor) ** np.arange(len(rewards))
        return float(gammas @ np.array(rewards))

    def _build_episodic_data(self) -> np.ndarray:
        with_history = (
            self.ratings.join(self.item_index_map, ["asin"])
            .withColumn(
                "user_history",
                collect_list("item_index").over(
                    W.partitionBy("reviewerID")
                    .orderBy("unixReviewTime")
                    .rowsBetween(W.unboundedPreceding, -1)
                ),
            )
            .filter(size("user_history") > 0)
        )

        min_length, max_length = self.threshold
        if min_length:
            with_history = with_history.filter(size("user_history") >= min_length - 1)
        if max_length:
            with_history = with_history.filter(size("user_history") <= max_length - 1)

        episodes_df = (
            with_history.withColumnRenamed("item_index", "action")
            .withColumn("reward", col("overall") - 3.0)
            .withColumn(
                "reward_episode",
                collect_list("reward").over(
                    W.partitionBy("reviewerID")
                    .orderBy("unixReviewTime")
                    .rowsBetween(W.currentRow, W.unboundedFollowing)
                ),
            )
            .withColumn("discount_factor", lit(self.discount_factor))
            .withColumn(
                "return", self._compute_return("reward_episode", "discount_factor")
            )
            .select("user_history", "action", "return")
        )

        episodic_samples = np.array(
            [
                (row["user_history"], row["action"], row["return"])
                for row in episodes_df.collect()
            ]
        )

        return episodic_samples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[List[int], int, float]:
        user_history, action, _return = self.data[idx]
        return user_history, action, _return


@dataclass
class PaddedNSortedUserHistoryBatch:
    data: torch.LongTensor
    lengths: torch.LongTensor


class UserItemEpisodeTrainLoader(DataLoader):
    padding_signal = -1

    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs, collate_fn=self.collate_function)

    @staticmethod
    def collate_function(
        batch: List[Tuple[List[int], int, float]],
    ) -> Tuple[PaddedNSortedUserHistoryBatch, torch.LongTensor, torch.FloatTensor]:
        batch_size = len(batch)
        user_history, action, _return = tuple(np.array(batch, dtype=object).T)

        padded_user_history, lengths = UserItemEpisodeTrainLoader.pad_sequence(
            user_history
        )
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        return (
            PaddedNSortedUserHistoryBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            torch.from_numpy(action.astype(np.int64)).view(batch_size, -1),
            torch.from_numpy(_return.astype(np.float32)).view(batch_size, -1),
        )

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
                        + UserItemEpisodeTrainLoader.padding_signal,
                    ]
                ).long()
                for item_seq in user_history
            ]
        )
        return padded, lengths

    import datetime


class AmazonReviewEvalDataset(Dataset):
    ratings_fetching_schema = StructType(
        [
            StructField("reviewerID", StringType()),
            StructField("reviewerName", StringType()),
            StructField("asin", StringType()),
            StructField("overall", FloatType()),
            StructField("vote", IntegerType()),
            StructField("reviewTime", StringType()),
            StructField("unixReviewTime", LongType()),
            StructField("verified", BooleanType()),
        ]
    )

    def __init__(
        self,
        data_path: str,
        category_name: str,
        spark: SparkSession,
        start_date: datetime.date,
        end_date: datetime.date,
        train_item_index_map: DataFrame,
        min_sequence_length: int = None,
        max_sequence_length: int = None,
    ):
        self.spark = spark
        self.category_name = category_name
        self.start_date = start_date
        self.end_date = end_date
        self.threshold = (min_sequence_length, max_sequence_length)

        self.ratings: DataFrame = self._get_ratings_df(
            os.path.join(data_path, f"{self.category_name.replace(' ', '_')}.json")
        )

        self.item_index_map: DataFrame = train_item_index_map
        self.data: np.ndarray = self._build_data()

    def _get_ratings_df(self, json_path):
        raw = self.spark.read.schema(self.ratings_fetching_schema).json(json_path)

        preprocessed = raw.withColumn(
            "timestamp", from_unixtime("unixReviewTime")
        ).filter(col("timestamp").cast("date").between(self.start_date, self.end_date))

        return preprocessed

    def _build_data(self) -> np.ndarray:
        with_history = (
            self.ratings.join(
                self.item_index_map, ["asin"]
            )  # filter unseen items in training
            .withColumn(
                "user_history",
                collect_list("item_index").over(
                    W.partitionBy("reviewerID")
                    .orderBy("unixReviewTime")
                    .rowsBetween(W.unboundedPreceding, -1)
                ),
            )
            .filter(size("user_history") > 0)
        )

        min_length, max_length = self.threshold
        if min_length:
            with_history = with_history.filter(size("user_history") >= min_length - 1)
        if max_length:
            with_history = with_history.filter(size("user_history") <= max_length - 1)

        episodes_df = (
            with_history.withColumn(
                "episode",
                collect_list("item_index").over(
                    W.partitionBy("reviewerID")
                    .orderBy("unixReviewTime")
                    .rowsBetween(W.currentRow, W.unboundedFollowing)
                ),
            )
            .withColumn(
                "relevance",
                collect_list(col("overall") - 3.0).over(
                    W.partitionBy("reviewerID")
                    .orderBy("unixReviewTime")
                    .rowsBetween(W.currentRow, W.unboundedFollowing)
                ),
            )
            .select("reviewerID", "user_history", "episode", "relevance")
        )

        eval_records = np.array(
            [
                (
                    row["reviewerID"],
                    row["user_history"],
                    row["episode"],
                    row["relevance"],
                )
                for row in episodes_df.collect()
            ]
        )

        return eval_records

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, List[int], int, float]:
        user_id, user_history, episode, relevance = self.data[idx]
        return user_id, user_history, episode, relevance


class UserItemEpisodeEvalLoader(DataLoader):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs, collate_fn=self.collate_function)

    @staticmethod
    def collate_function(
        batch: List[Tuple[str, List[int], int, float]],
    ) -> Tuple[
        List[str],
        PaddedNSortedUserHistoryBatch,
        List[torch.LongTensor],
        List[torch.FloatTensor],
    ]:
        user_id, user_history, episode, relevance = tuple(
            np.array(batch, dtype=object).T
        )

        padded_user_history, lengths = UserItemEpisodeTrainLoader.pad_sequence(
            user_history
        )
        sorted_lengths, sorted_idx = lengths.sort(0, descending=True)

        return (
            list(user_id),
            PaddedNSortedUserHistoryBatch(
                data=padded_user_history[sorted_idx],
                lengths=sorted_lengths,
            ),
            [torch.LongTensor(seq) for seq in episode],
            [torch.FloatTensor(seq) for seq in relevance],
        )
