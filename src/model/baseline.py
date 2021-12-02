from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame
from pyspark.sql import Window as W
from pyspark.sql.functions import (
    avg,
    col,
    collect_list,
    explode,
    lit,
    monotonically_increasing_id,
    slice,
    struct,
    udf,
)
from pyspark.sql.types import FloatType, StructField, StructType


class CollaborativeFiltering:
    def __init__(self, **model_params) -> None:
        params_validated = {
            k: v
            for k, v in model_params.items()
            if k not in ("userCol", "itemCol", "ratingCol", "implicitPrefs")
        }
        self.als = ALS(
            userCol="user_int_id",
            itemCol="item_int_id",
            ratingCol="overall",
            implicitPrefs=False,
            **params_validated
        )

        self.model = None
        self.user_id_map = self.item_id_map = None
        self.rating_logs = None

    def train(self, train_set: DataFrame):
        self.user_id_map, self.item_id_map = self._build_integer_id_maps(train_set)

        train_set_preprocessed = (
            train_set.join(self.user_id_map, ["reviewerID"])
            .join(self.item_id_map, ["asin"])
            .select("user_int_id", "item_int_id", "overall")
        )

        self.rating_logs = train_set.select("user_int_id", "item_int_id").distinct()

        self.model = self.als.fit(train_set_preprocessed)

    def eval(self, eval_set: DataFrame, K: int) -> Tuple[DataFrame, Dict[str, float]]:
        assert self.model, "Model not yet been trained"

        eval_set_preprocessed = (
            eval_set.join(self.user_id_map, ["reviewerID"])
            .join(self.item_id_map, ["asin"])
            .groupBy("user_int_id")
            .agg(collect_list(struct("item_int_id", "overall")).alias("actuals"))
            .select("user_int_id", "actuals")
        )

        recommendations_df = (
            self.model.recommendForUserSubset(eval_set_preprocessed, -1)
            .withColumn("recommendation", explode("recommendations"))
            .select(
                "user_int_id", "recommendation.item_int_id", "recommendation.rating"
            )
            .join(
                self.rating_logs.withColumn("flag", lit(True)),
                ["user_int_id", "item_int_id"],
                "left",
            )
            .filter(col("flag").isNull())
            .withColumn(
                "recommendations",
                collect_list(struct("item_int_id", "rating")).over(
                    W.partitionBy("user_int_id")
                    .orderBy(col("rating").desc())
                    .rowsBetween(W.unboundedPreceding, W.unboundedFollowing)
                ),
            )
            .withColumn("recommendations", slice("recommendations", 1, K))
            .select("user_int_id", "recommendations")
            .distinct()
        )

        eval_df = recommendations_df.join(
            eval_set_preprocessed, ["user_int_id"]
        ).withColumn(
            "eval_log", self._eval_recommendations("actuals", "recommendations")
        )

        summary = (
            eval_df.select(
                avg("eval_log.precision_at_K").alias("precision_at_K"),
                avg("eval_log.recall_at_K").alias("recall_at_K"),
                avg("eval_log.ndcg_at_K").alias("ndcg_at_K"),
            )
            .first()
            .asDict()
        )

        return eval_df, summary

    @staticmethod
    @udf(
        StructType(
            [
                StructField("precision_at_K", FloatType()),
                StructField("recall_at_K", FloatType()),
                StructField("ndcg_at_K", FloatType()),
            ]
        )
    )
    def _eval_recommendations(
        actuals: List[Dict[str, Union[int, float]]],
        recommendations: List[Dict[str, Union[int, float]]],
    ) -> Dict[str, Optional[float]]:
        positive_items = set(
            log["item_int_id"] for log in actuals if log["overall"] > 3.0
        )
        recommended_items = set(
            rec["item_int_id"] for rec in recommendations if rec["rating"] > 3.0
        )
        intersections = positive_items & recommended_items

        positive_rels = [
            log["overall"] - 3.0 for log in actuals if log["overall"] > 3.0
        ]
        _coefs = 1.0 / np.log2(np.arange(2, len(positive_rels) + 2))
        idcg = _coefs @ np.sort(positive_rels)[::-1]
        relevance_book = {log["item_int_id"]: log["overall"] - 3.0 for log in actuals}
        _coefs = 1.0 / np.log2(np.arange(2, len(recommendations) + 2))
        dcg = _coefs @ np.array(
            [relevance_book.get(rec["item_int_id"]) or 0.0 for rec in recommendations]
        )

        precision = (
            len(intersections) / len(recommended_items)
            if len(recommended_items) > 0
            else None
        )
        recall = (
            len(intersections) / len(positive_items)
            if len(positive_items) > 0
            else None
        )
        ndcg = float(dcg / idcg) if idcg > 0 else None

        return {
            "precision_at_K": precision,
            "recall_at_K": recall,
            "ndcg_at_K": ndcg,
        }

    def _build_integer_id_maps(
        self, train_set: DataFrame
    ) -> Tuple[DataFrame, DataFrame]:
        user_id_map = (
            train_set.select("reviewerID")
            .distinct()
            .coalesce(1)
            .orderBy("reviewerID")
            .withColumn("user_int_id", monotonically_increasing_id())
        )

        item_id_map = (
            train_set.select("asin")
            .distinct()
            .coalesce(1)
            .orderBy("asin")
            .withColumn("item_int_id", monotonically_increasing_id())
        )

        return user_id_map, item_id_map
