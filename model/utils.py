import numpy as np


def calculate_rank(score, target, filter_list):
    target_score = score[target]
    score[filter_list] = target_score - 1
    rank = np.sum(score > target_score) + np.sum(score == target_score) // 2 + 1
    return rank


def metrics(rank_list):
    mr = np.mean(rank_list)
    mrr = np.mean(1 / rank_list)
    hit10 = np.sum(rank_list < 11) / len(rank_list)
    hit3 = np.sum(rank_list < 4) / len(rank_list)
    hit1 = np.sum(rank_list < 2) / len(rank_list)
    return mr, mrr, hit10, hit3, hit1
