import datetime
import os
from typing import List, Tuple

import numpy as np
import torch
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, collect_list, from_unixtime, size
from pyspark.sql.types import (
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

from train import PaddedNSortedUserHistoryBatch, UserItemEpisodeTrainLoader


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
            with_history = with_history.filter(size("user_history") >= min_length)
        if max_length:
            with_history = with_history.filter(size("user_history") <= max_length)

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
                collect_list("rating").over(
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
        List[str], PaddedNSortedUserHistoryBatch, torch.LongTensor, torch.FloatTensor
    ]:
        batch_size = len(batch)
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
            torch.from_numpy(episode.astype(np.int64)).view(batch_size, -1),
            torch.from_numpy(relevance.astype(np.float32)).view(batch_size, -1),
        )
