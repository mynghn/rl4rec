from collections import defaultdict
from typing import DefaultDict, Dict, List

import numpy as np
import torch
from torch.nn.modules.loss import KLDivLoss
from tqdm import tqdm

from ..dataset.retailrocket import RetailrocketDataLoader
from ..model.agent import TopKOfflineREINFORCE


def evaluate_agent(
    agent: TopKOfflineREINFORCE,
    eval_loader: RetailrocketDataLoader,
    discount_factor: float,
    device: torch.device = torch.device("cpu"),
    debug: bool = False,
) -> Dict[str, float]:
    agent = agent.to(device)
    agent.eval()

    kl_div_loss = KLDivLoss(reduction="sum", log_target=True)

    expected_return_cumulative = 0.0
    kl_div_sum = 0.0
    precision_cumulatvie = 0.0
    recall_cumulative = 0.0
    ndcg_cumulative = 0.0
    hit = 0
    users_hit = set()

    iter_cnt = 0
    users_total = set()
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="eval"):
            batch = eval_loader.to(batch=batch, device=device)
            batch_size = len(batch["user_id"])

            beta_state = agent.beta_state_network(
                user_history=batch["user_history"],
                user_feature_index=batch.get("user_feature_index"),
                item_feature_index=batch.get("item_feature_index"),
            )
            pi_state = agent.pi_state_network(
                user_history=batch["user_history"],
                user_feature_index=batch.get("user_feature_index"),
                item_feature_index=batch.get("item_feature_index"),
            )

            # 1. Expected return of Agent's policy over samples from eval data that follows behavior policy
            item_index_tensor = (
                torch.LongTensor([seq[0] for seq in batch["item_index_episode"]])
                .view(batch_size, -1)
                .to(device)
            )
            corrected_return = agent.get_corrected_return(
                beta_state=beta_state,
                pi_state=pi_state,
                item_index=item_index_tensor,
                episodic_return=batch["return"],
            )
            expected_return_cumulative += corrected_return.sum().cpu().item()

            # 2. KL Divergence between Agent's policy & former behavior policy
            log_action_policy_probs = agent.action_policy.log_probs(pi_state)
            log_behavior_policy_probs = agent.behavior_policy.log_probs(beta_state)

            kl_div_sum += kl_div_loss(
                log_action_policy_probs, log_behavior_policy_probs
            )

            # 3. Compute metrics from traditional RecSys domain
            recommended_item_indices, _ = agent.get_top_recommendations(
                state=pi_state, M=agent.K
            )

            for user_id, actual_seq, reward_seq, recommendations in zip(
                batch["user_id"],
                batch["item_index_episode"],
                batch["reward_episode"],
                recommended_item_indices,
            ):
                recommendation_list = recommendations.tolist()

                actual_item_set = set(actual_seq)
                recommendation_set = set(recommendation_list)

                true_positive = len(actual_item_set & recommendation_set)
                if true_positive > 0:
                    hit += 1
                    users_hit.add(user_id)
                precision = true_positive / len(recommendation_set)
                recall = true_positive / len(actual_item_set)
                ndcg = compute_ndcg(
                    recommendations=recommendation_list,
                    relevance_book=build_relevance_book(
                        item_sequence=actual_seq,
                        reward_sequence=reward_seq,
                        discount_factor=discount_factor,
                    ),
                )

                precision_cumulatvie += precision
                recall_cumulative += recall
                ndcg_cumulative += ndcg

                if debug:
                    print(f"User: {user_id}")
                    print(f"1. Precision: {precision:2.%}")
                    print(f"1. Recall: {recall:2.%}")
                    print(f"1. nDCG: {ndcg:2.%}")
                    print("=" * 20)

            iter_cnt += batch_size
            users_total |= set(batch["user_id"])

    return {
        "E[Return]": expected_return_cumulative / iter_cnt,
        "KL-Divergence(Pi|Beta)": kl_div_sum / iter_cnt,
        f"Precision at {agent.K}": precision_cumulatvie / iter_cnt,
        f"Recall at {agent.K}": recall_cumulative / iter_cnt,
        f"nDCG at {agent.K}": ndcg_cumulative / iter_cnt,
        "Hit Rate": hit / iter_cnt,
        "User Hit Rate": len(users_hit) / len(users_total),
    }


def compute_ndcg(
    recommendations: List[int], relevance_book: DefaultDict[int, float]
) -> float:
    K = len(recommendations)
    coefs = torch.ones(K) / torch.arange(2, K + 2).log2()

    sorted_relevance_by_K = torch.FloatTensor(
        sorted(relevance_book.values(), reverse=True)[:K]
    )
    if sorted_relevance_by_K.size(0) < K:
        _pad = torch.zeros(K - sorted_relevance_by_K.size(0))
        sorted_relevance_by_K = torch.cat((sorted_relevance_by_K, _pad))
    idcg = (coefs @ sorted_relevance_by_K).item()

    recommended_relevance = torch.FloatTensor(
        [relevance_book[item_idx] for item_idx in recommendations]
    )
    if recommended_relevance.size(0) < K:
        _pad = torch.zeros(K - recommended_relevance.size(0))
        recommended_relevance = torch.cat((recommended_relevance, _pad))
    dcg = (coefs @ recommended_relevance).item()

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg


def build_relevance_book(
    item_sequence: List[int], reward_sequence: List[float], discount_factor: float
) -> DefaultDict[int, float]:
    assert len(item_sequence) == len(
        reward_sequence
    ), "Item and reward sequence length should match."

    gammas = (1.0 - discount_factor) ** np.arange(len(item_sequence))
    relevance_book = defaultdict(float)
    for gamma, item_index, reward in zip(gammas, item_sequence, reward_sequence):
        relevance_book[item_index] += reward * gamma

    return relevance_book
