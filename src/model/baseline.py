from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
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
from torch.nn.functional import softmax
from torch.nn.modules.rnn import PackedSequence

from ..model.nn import StateTransitionNetwork


class GRU4Rec(nn.Module):
    def __init__(
        self,
        n_items: int,
        hidden_size: int,
        n_gru_layers: int = 1,
        dropout: int = 0.4,
        user_action_embedding_dim: int = -1,
        n_actions: int = None,
        padding_signal: int = None,
    ):
        super(GRU4Rec, self).__init__()

        self.gru_layer = StateTransitionNetwork(
            n_items=n_items,
            hidden_size=hidden_size,
            num_layers=n_gru_layers,
            dropout=dropout,
            user_action_embedding_dim=user_action_embedding_dim,
            n_actions=n_actions,
            padding_signal=padding_signal,
        )
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=n_items),
            nn.Tanh(),
        )

    def forward(self, pack_padded_histories: PackedSequence) -> torch.FloatTensor:
        out, lengths = self.gru_layer(pack_padded_histories)
        out = self.output_layer(out)
        return out, lengths

    # Implement TOP1 loss from the paper
    def top1_loss(
        self, other_logits: torch.FloatTensor, relevant_logit: torch.FloatTensor
    ) -> torch.FloatTensor:
        return torch.mean(
            torch.sigmoid(
                other_logits
                - relevant_logit.unsqueeze(1).expand(-1, other_logits.size(1))
            )
            + torch.sigmoid(other_logits ** 2),
            dim=1,
        )

    def recommend(
        self, pack_padded_histories: PackedSequence, K: int
    ) -> List[List[List[int]]]:
        out, lengths = self.forward(pack_padded_histories)
        recommendations = []
        for b in range(lengths.size(0)):
            ep_recom = []
            for i in range(lengths[b]):
                sorted_indices = out[b, i, :].view(-1).argsort(descending=True)
                items = sorted_indices[:K]
                ep_recom.append(items)
            recommendations.append(ep_recom)
        return recommendations

    def get_probs(self, pack_padded_histories: PackedSequence) -> torch.FloatTensor:
        logits, lengths = self(pack_padded_histories)
        return softmax(logits), lengths

    def get_corrected_batch_return(
        self,
        model_probs: torch.FloatTensor,
        behavior_policy_probs: torch.FloatTensor,
        lengths: torch.LongTensor,
        item_index: Sequence[Sequence[float]],
        return_at_t: Sequence[Sequence[float]],
    ) -> float:
        batch_size = len(lengths)
        batch_return_cumulated = 0.0
        for b in range(batch_size):
            ep_return = 0.0
            for t in range(1, lengths[b] + 1):
                item_index_in_episode = item_index[b][t:]
                importance_weight = model_probs[b, t - 1, item_index_in_episode] @ (
                    1 / behavior_policy_probs[b, t - 1, item_index_in_episode]
                )
                ep_return += importance_weight.cpu().item() * return_at_t[b][t]
            batch_return_cumulated += ep_return / lengths[b].cpu().item()
        return batch_return_cumulated / batch_size


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
