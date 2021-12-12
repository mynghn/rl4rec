from collections import defaultdict
from itertools import combinations
from typing import DefaultDict, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
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

from ..command.eval import build_relevance_book, compute_ndcg
from ..dataset.retailrocket import RetailrocketDataset
from ..model.agent import TopKOfflineREINFORCE


class JaccardSimilarity:
    def __init__(self):
        self.users_by_item_book: DefaultDict[int, Set[str]] = defaultdict(set)
        self.model: Dict[str, Dict[str, float]] = {}

    def train(self, train_set: RetailrocketDataset):
        for user, items in train_set.lifetime_items_book.items():
            for item in items:
                self.users_by_item_book[item].add(user)

        for item1, item2 in combinations(self.users_by_item_book.keys(), 2):
            if item1 not in self.model.keys():
                self.model[item1] = {}
            if item2 not in self.model.keys():
                self.model[item2] = {}

            item1_users = self.users_by_item_book[item1]
            item2_users = self.users_by_item_book[item2]

            intersections = len(item1_users & item2_users)
            unions = len(item1_users | item2_users)

            self.model[item1][item2] = self.model[item2][item1] = intersections / unions

    def recommend(self, input_items: List[int], K: int) -> List[int]:
        assert self.model, "Model not been built yet."

        score_book = defaultdict(float)
        for item_in in input_items:
            if item_in in self.model.keys():
                for item_out, sim in self.model[item_in].items():
                    score_book[item_out] += sim

        top_K = sorted(score_book.items(), key=lambda item: item[1], reverse=True)[:K]

        return [item for item, _ in top_K]

    def get_item_probs(self, input_items: List[int]) -> DefaultDict[int, float]:
        assert self.model, "Model not been built yet."

        score_book = defaultdict(float)
        for item_in in input_items:
            if item_in in self.model.keys():
                for item_out, sim in self.model[item_in].items():
                    score_book[item_out] += sim

        ordered = sorted(score_book.items(), key=lambda item: item[1], reverse=True)
        items = [item for item, _ in ordered]
        logits = [logit for _, logit in ordered]
        probs = self.softmax(logits)

        item_prob_book = defaultdict(float)
        for item, prob in zip(items, probs):
            item_prob_book[item] = prob

        return item_prob_book

    def softmax(self, logits: List[float]) -> List[float]:
        numer = np.exp(logits)
        denom = np.sum(numer)
        return list(numer / denom)

    def eval(
        self,
        agent: TopKOfflineREINFORCE,
        eval_set: RetailrocketDataset,
        K: int,
        discount_factor: float,
    ) -> Dict[str, float]:
        agent.to("cpu")
        agent.eval()

        expected_return_cumulative = 0.0
        precision_cumulatvie = 0.0
        recall_cumulative = 0.0
        ndcg_cumulative = 0.0
        hit = 0
        users_hit = set()

        iter_cnt = 0
        users_total = set()
        with torch.no_grad():
            for idx, data in enumerate(eval_set):
                (
                    user_id,
                    user_history,
                    actual_seq,
                    reward_seq,
                    episodic_return,
                ) = data

                index_in_history = eval_set.df.iloc[idx]["event_index_in_user_history"]
                past_item_sequence = eval_set.lifetime_items_book[user_id][
                    :index_in_history
                ]
                # 1. Expected return of model over samples from eval data that follows behavior policy
                user_history_tensor = torch.LongTensor(user_history).view(1, 1)
                beta_state = agent.beta_state_network(user_history=user_history_tensor)
                item_index_tensor = torch.LongTensor(actual_seq[:1]).view(1, 1)
                behavior_policy_prob = torch.exp(
                    agent.behavior_policy(beta_state, item_index_tensor)
                ).item()
                model_prob = self.get_item_probs(past_item_sequence)[actual_seq[0]]
                corrected_return = model_prob / behavior_policy_prob * episodic_return
                expected_return_cumulative += corrected_return

                # 2. Compute metrics from traditional RecSys domain
                recommendation_list = self.recommend(past_item_sequence, K)

                actual_item_set = set(actual_seq)
                recommendation_set = set(recommendation_list)

                true_positive = len(actual_item_set & recommendation_set)
                if true_positive > 0:
                    hit += 1
                    users_hit.add(user_id)
                precision_cumulatvie += true_positive / len(recommendation_set)
                recall_cumulative += true_positive / len(actual_item_set)
                ndcg_cumulative += compute_ndcg(
                    recommendations=recommendation_list,
                    relevance_book=build_relevance_book(
                        item_sequence=actual_seq,
                        reward_sequence=reward_seq,
                        discount_factor=discount_factor,
                    ),
                )

                iter_cnt += 1
                users_total.add(user_id)

        return {
            "E[Return]": expected_return_cumulative / iter_cnt,
            f"Precision at {K}": precision_cumulatvie / iter_cnt,
            f"Recall at {K}": recall_cumulative / iter_cnt,
            f"nDCG at {K}": ndcg_cumulative / iter_cnt,
            "Hit Rate": hit / iter_cnt,
            "User Hit Rate": len(users_hit) / len(users_total),
        }


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
            **params_validated,
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

        self.rating_logs = train_set_preprocessed.select(
            "user_int_id", "item_int_id"
        ).distinct()

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
            self.model.recommendForUserSubset(
                eval_set_preprocessed, self.item_id_map.count()
            )
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
            .filter(col("rating") > 3.0)
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
        recommended_items = set(rec["item_int_id"] for rec in recommendations)
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
