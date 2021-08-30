import numpy as np


def min_normalizer(raw_score_history):
    return raw_score_history / np.nanmin(raw_score_history)
