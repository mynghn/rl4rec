import datetime
import os
from typing import List, Optional

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import (
    col,
    collect_list,
    from_unixtime,
    lit,
    monotonically_increasing_id,
    regexp_extract,
    regexp_replace,
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
from torch.utils.data import Dataset


class AmazonReviewDataset(Dataset):
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
        start_date: datetime.date,
        end_date: datetime.date,
        discount_factor: float,
    ):
        self.spark = spark
        self.category_name = category_name
        self.start_date = start_date
        self.end_date = end_date
        self.discount_factor = discount_factor

        self.ratings: DataFrame = self._get_ratings_df(
            os.path.join(data_path, f"{self.category_name.replace(' ', '_')}.json")
        )
        self.item_metadata: DataFrame = self._get_metadata_df(
            os.path.join(data_path, f"meta_{self.category_name.replace(' ', '_')}.json")
        )

        self.item_index_map: DataFrame = self._build_item_index_map()
        self.data: np.ndarray = self._build_episodic_data()

    def _get_ratings_df(self, json_path):
        raw = self.spark.read.schema(self.ratings_fetching_schema).json(json_path)

        preprocessed = raw.withColumn(
            "timestamp", from_unixtime("unixReviewTime")
        ).filter(col("timestamp").cast("date").between(self.start_date, self.end_date))

        return preprocessed

    def _get_metadata_df(self, json_path):
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
            .orderBy("asin")
            .withColumn("item_index", monotonically_increasing_id())
        )

    @staticmethod
    @udf(FloatType())
    def _compute_return(rewards: List[float], discount_factor: float) -> float:
        gammas = (1.0 - discount_factor) ** np.arange(len(rewards))
        return float(gammas @ np.array(rewards))

    def _build_episodic_data(self) -> np.ndarray:
        episodes_df = (
            self.ratings.join(self.item_index_map, ["asin"])
            .withColumn(
                "user_history",
                collect_list("item_index").over(
                    W.partitionBy("reviewerID")
                    .orderBy("unixReviewTime")
                    .rowsBetween(W.unboundedPreceding, W.currentRow)
                ),
            )
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
            .select("user_history", "return")
        )

        episodic_samples = np.array(
            [(row["user_history"], row["return"]) for row in episodes_df.collect()]
        )

        return episodic_samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        user_history, _return = self.data[idx]
        return user_history, _return
