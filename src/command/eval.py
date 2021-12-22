from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Union

import numpy as np
import torch
from torch.nn.modules.loss import KLDivLoss
from tqdm import tqdm

from ..dataset.retailrocket import RetailrocketEpisodeLoader
from ..model.agent import TopKOfflineREINFORCE
from ..model.baseline import GRU4Rec
from ..model.policy import BehaviorPolicy


def evaluate_recommender(
    model: Union[TopKOfflineREINFORCE, GRU4Rec],
    eval_loader: RetailrocketEpisodeLoader,
    discount_factor: float,
    device: torch.device = torch.device("cpu"),
    debug: bool = False,
    K: int = None,
    behavior_policy: Optional[BehaviorPolicy] = None,
) -> Dict[str, float]:
    model = model.to(device)
    model.eval()
    kl_div_loss = KLDivLoss(reduction="batchmean", log_target=True)

    expected_return_cumulated = 0.0
    return_cumulated = 0.0
    kl_div_cumulated = 0.0
    precision_cumulated = 0.0
    recall_cumulated = 0.0
    ndcg_cumulated = 0.0
    hit = 0
    users_hit = set()

    iter_cnt = 0
    users_total = set()

    with torch.no_grad():
        if isinstance(model, TopKOfflineREINFORCE):
            K = model.K
            ep_cnt = 0
            for batch in tqdm(eval_loader, desc="eval"):
                batch = eval_loader.to(batch=batch, device=device)

                return_cumulated += sum(
                    [sum(ep[1:]) / len(ep[1:]) for ep in batch["return_at_t"]]
                )

                # Build States
                beta_state, _ = model.behavior_policy.struct_state(
                    batch["pack_padded_histories"]
                )

                pi_state, lengths = model.pi_state_network(
                    batch["pack_padded_histories"]
                )

                batch_size = len(batch["item_episodes"])
                for b in range(batch_size):
                    ep_len = lengths[b] + 1
                    # 1. Expected return of model's policy over samples from eval data that follows behavior policy
                    expected_return_cumulated += model.get_corrected_episodic_return(
                        pi_state=pi_state[b, : ep_len - 1, :].view(ep_len - 1, -1),
                        beta_state=beta_state[b, : ep_len - 1, :].view(ep_len - 1, -1),
                        item_index=(
                            torch.LongTensor(list(batch["item_episodes"][b])[1:ep_len])
                            .view(ep_len - 1, 1)
                            .to(device)
                        ),
                        return_at_t=(
                            torch.FloatTensor(list(batch["return_at_t"][b])[1:ep_len])
                            .view(ep_len - 1, 1)
                            .to(device)
                        ),
                    )

                    # 2. KL Divergence between Agent's policy & former behavior policy
                    behavior_policy_log_probs = model.behavior_policy(
                        state=beta_state[b, : ep_len - 1, :].view(ep_len - 1, -1),
                        item_index=(
                            torch.LongTensor(batch["item_episodes"][b][1:ep_len])
                            .view(ep_len - 1, 1)
                            .to(device)
                        ),
                    )
                    model_log_probs = model.action_policy_head(
                        state=pi_state[b, : ep_len - 1, :].view(ep_len - 1, -1),
                        item_index=(
                            torch.LongTensor(batch["item_episodes"][b][1:ep_len])
                            .view(ep_len - 1, 1)
                            .to(device)
                        ),
                    )

                    kl_div_cumulated += (
                        kl_div_loss(model_log_probs, behavior_policy_log_probs)
                        .cpu()
                        .item()
                    )

                    # 3. Compute metrics from traditional RecSys domain
                    recommended_item_indices, _ = model.get_top_recommendations(
                        pi_state[b, : ep_len - 1, :].view(ep_len - 1, -1), M=K
                    )

                    ep_cnt += 1

                    for t, user_id, recommendations in zip(
                        range(1, ep_len), batch["user_id"], recommended_item_indices
                    ):
                        recommendation_list = recommendations.tolist()
                        actual_seq = list(batch["item_episodes"][b])[t:]
                        reward_seq = list(batch["return_at_t"][b])[t:]
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

                        precision_cumulated += precision
                        recall_cumulated += recall
                        ndcg_cumulated += ndcg

                        if debug:
                            print(f"User: {user_id}")
                            print(f"1. Precision: {precision:2.%}")
                            print(f"1. Recall: {recall:2.%}")
                            print(f"1. nDCG: {ndcg:2.%}")
                            print("=" * 20)

                        iter_cnt += 1

                users_total |= set(batch["user_id"])

            expected_return = expected_return_cumulated / ep_cnt
            kl_div = kl_div_cumulated / ep_cnt

        elif isinstance(model, GRU4Rec):
            assert behavior_policy is not None and K is not None
            batch_cnt = 0
            ep_cnt = 0
            for batch in tqdm(eval_loader, desc="eval"):
                batch = eval_loader.to(batch=batch, device=device)

                return_cumulated += sum(
                    [sum(ep[1:]) / len(ep[1:]) for ep in batch["return_at_t"]]
                )

                # 1. Expected return of model's policy over samples from eval data that follows behavior policy
                batch_model_log_probs, _ = model.log_probs(
                    batch["pack_padded_histories"]
                )
                beta_state, lengths = behavior_policy.struct_state(
                    batch["pack_padded_histories"]
                )
                batch_behavior_policy_log_probs = behavior_policy.log_probs(beta_state)
                expected_return_cumulated += model.get_corrected_batch_return(
                    model_log_probs=batch_model_log_probs,
                    behavior_policy_log_probs=batch_behavior_policy_log_probs,
                    lengths=lengths,
                    item_index=batch["item_episodes"],
                    return_at_t=batch["return_at_t"],
                )

                # Recommendations
                recommended_item_indices = model.recommend(
                    batch["pack_padded_histories"], K
                )

                batch_size = len(batch["item_episodes"])
                for b in range(batch_size):
                    ep_len = lengths[b] + 1
                    # 2. KL Divergence between Agent's policy & former behavior policy
                    behavior_policy_log_probs = batch_behavior_policy_log_probs[
                        b, : ep_len - 1, batch["item_episodes"][b][1:ep_len]
                    ].view(ep_len - 1, -1)
                    model_log_probs = batch_model_log_probs[
                        b, : ep_len - 1, batch["item_episodes"][b][1:ep_len]
                    ].view(ep_len - 1, -1)
                    kl_div_cumulated += (
                        kl_div_loss(model_log_probs, behavior_policy_log_probs)
                        .cpu()
                        .item()
                    )

                    ep_cnt += 1

                    # 3. Compute metrics from traditional RecSys domain
                    for t, user_id, recommendations in zip(
                        range(1, ep_len), batch["user_id"], recommended_item_indices[b]
                    ):
                        recommendation_list = recommendations.tolist()
                        actual_seq = list(batch["item_episodes"][b])[t:]
                        reward_seq = list(batch["return_at_t"][b])[t:]
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

                        precision_cumulated += precision
                        recall_cumulated += recall
                        ndcg_cumulated += ndcg

                        if debug:
                            print(f"User: {user_id}")
                            print(f"1. Precision: {precision:2.%}")
                            print(f"1. Recall: {recall:2.%}")
                            print(f"1. nDCG: {ndcg:2.%}")
                            print("=" * 20)

                        iter_cnt += 1

                users_total |= set(batch["user_id"])
                batch_cnt += 1

            expected_return = expected_return_cumulated / batch_cnt
            kl_div = kl_div_cumulated / ep_cnt
        else:
            raise TypeError("Unregistered recommender type encountered.")

    return {
        "E_pi[Return]": expected_return,
        "E_beta[Return]": return_cumulated / ep_cnt,
        "KL-Divergence(Pi|Beta)": kl_div,
        f"Precision at {K}": precision_cumulated / iter_cnt,
        f"Recall at {K}": recall_cumulated / iter_cnt,
        f"nDCG at {K}": ndcg_cumulated / iter_cnt,
        "Hit Rate": hit / iter_cnt,
        "User Hit Rate": len(users_hit) / len(users_total),
        "# of Users Tested": len(users_total),
        "# of Recommendations Tested": iter_cnt,
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
