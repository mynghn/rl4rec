from typing import Dict

import torch
from tqdm import tqdm

from ..dataset.eval import UserItemEpisodeEvalLoader
from ..model.agent import TopKOfflineREINFORCE


def evaluate_agent(
    agent: TopKOfflineREINFORCE,
    eval_loader: UserItemEpisodeEvalLoader,
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

    agent.eval()
    for (
        user_id_list,
        prev_item_sequence,
        item_sequence_list,
        relevance_sequence_list,
    ) in tqdm(eval_loader, desc="eval"):
        user_state = agent.state_network(prev_item_sequence)

        recommended_items, _ = agent.recommend(user_state)

        for user_id, actual_seq, relevance_seq, recommendations in zip(
            user_id_list, item_sequence_list, relevance_sequence_list, recommended_items
        ):
            intersection = set(actual_seq.tolist()) & set(recommendations.tolist())
            if len(intersection) > 0:
                hit += 1
                users_hit.add(user_id)

            precision = len(intersection) / len(recommendations)
            recall = len(intersection) / len(actual_seq)
            ndcg = compute_ndcg(recommendations, actual_seq, relevance_seq)

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
        users_total |= set(user_id_list)

    return {
        f"Precision at {K}": sum(precision_log) / cnt,
        f"Recall at {K}": sum(recall_log) / cnt,
        f"nDCG at {K}": sum(ndcg_log) / cnt,
        "Hit Rate": hit / cnt,
        "User Hit Rate": len(users_hit) / len(users_total),
    }


def compute_ndcg(
    recommendations: torch.LongTensor,
    actual_sequence: torch.LongTensor,
    relevance_sequence: torch.FloatTensor,
) -> float:
    positive_relevance = relevance_sequence[relevance_sequence > 0]
    sorted_relevance, _ = positive_relevance.sort(descending=True)

    n_positive_items = len(positive_relevance)
    _coefs = torch.ones(n_positive_items) / torch.arange(2, n_positive_items + 2).log2()
    idcg = (_coefs @ sorted_relevance).item()

    relevance_book = {
        actual_sequence[i].item(): relevance_sequence[i].item()
        for i in range(len(actual_sequence))
    }
    recommended_relevance = torch.FloatTensor(
        [relevance_book.get(item_idx.item()) or 0.0 for item_idx in recommendations]
    )
    K = len(recommendations)
    _coefs = torch.ones(K) / torch.arange(2, K + 2).log2()
    dcg = (_coefs @ recommended_relevance).item()

    return dcg / idcg if idcg > 0 else 0.0
