from collections import defaultdict
from typing import DefaultDict, Dict, List

import torch
from tqdm import tqdm

from ..dataset.cikim19 import CIKIM19DataLoader
from ..model.agent import TopKOfflineREINFORCE


def evaluate_agent(
    agent: TopKOfflineREINFORCE,
    eval_loader: CIKIM19DataLoader,
    device: torch.device = torch.device("cpu"),
    debug: bool = True,
) -> Dict[str, float]:
    K = agent.K
    precision_log = []
    recall_log = []
    ndcg_log = []
    cnt = 0
    hit = 0
    users_total = set()
    users_hit = set()

    agent = agent.to(device)

    agent.eval()
    for batch_dict in tqdm(eval_loader, desc="eval"):
        batch_dict = eval_loader.to(batch=batch_dict, device=device)

        if (
            agent.state_network.user_feature_enabled
            and agent.state_network.item_feature_enabled
        ):
            state = agent.state_network(
                user_history=batch_dict["user_history"],
                user_feature_index=batch_dict.get("user_feature_index"),
                item_feature_index=batch_dict.get("item_feature_index"),
            )
        elif agent.state_network.user_feature_enabled:
            state = agent.state_network(
                user_history=batch_dict["user_history"],
                user_feature_index=batch_dict.get("user_feature_index"),
            )
        elif agent.state_network.item_feature_enabled:
            state = agent.state_network(
                user_history=batch_dict["user_history"],
                item_feature_index=batch_dict.get("item_feature_index"),
            )
        else:
            state = agent.state_network(user_history=batch_dict["user_history"])

        recommended_item_indices, _ = agent.recommend(state)

        for user_id, actual_seq, reward_seq, recommendations in zip(
            batch_dict["user_id"],
            batch_dict["item_index_episode"],
            batch_dict["reward_episode"],
            recommended_item_indices,
        ):
            actual_item_set = set(actual_seq)
            recommendation_set = set(recommendations.tolist())

            intersections = actual_item_set & recommendation_set
            n_intersections = len(intersections)
            if n_intersections > 0:
                hit += 1
                users_hit.add(user_id)

            precision = n_intersections / len(recommendation_set)
            recall = n_intersections / len(actual_item_set)
            ndcg = compute_ndcg(
                recommendations=recommendations,
                relevance_book=build_relevance_book(
                    item_sequence=actual_seq, reward_sequence=reward_seq
                ),
            )

            precision_log.append(precision)
            recall_log.append(recall)
            ndcg_log.append(ndcg)

            if debug:
                print(f"User: {user_id}")
                print(f"1. Precision: {precision:2.%}")
                print(f"1. Recall: {recall:2.%}")
                print(f"1. nDCG: {ndcg:2.%}")
                print("=" * 20)

        cnt += eval_loader.batch_size
        users_total |= set(user_id)

    return {
        f"Precision at {K}": sum(precision_log) / cnt,
        f"Recall at {K}": sum(recall_log) / cnt,
        f"nDCG at {K}": sum(ndcg_log) / cnt,
        "Hit Rate": hit / cnt,
        "User Hit Rate": len(users_hit) / len(users_total),
    }


def compute_ndcg(
    recommendations: torch.LongTensor, relevance_book: DefaultDict[int, float]
) -> float:
    sorted_relevance = torch.FloatTensor(sorted(relevance_book.values(), reverse=True))

    n_items = len(sorted_relevance)
    _coefs = torch.ones(n_items) / torch.arange(2, n_items + 2).log2()
    idcg = (_coefs @ sorted_relevance).item()

    recommended_relevance = torch.FloatTensor(
        [relevance_book[item_idx.item()] for item_idx in recommendations]
    )
    K = len(recommendations)
    _coefs = torch.ones(K) / torch.arange(2, K + 2).log2()
    dcg = (_coefs @ recommended_relevance).item()

    return dcg / idcg if idcg > 0 else 0.0


def build_relevance_book(
    item_sequence: List[int], reward_sequence: List[float]
) -> DefaultDict[int, float]:
    assert len(item_sequence) == len(
        reward_sequence
    ), "Item and reward sequence length should match."

    relevance_book = defaultdict(float)
    for item_indexed, reward in zip(item_sequence, reward_sequence):
        relevance_book[item_indexed] += reward

    return relevance_book
