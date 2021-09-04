import numpy as np


def max_normalizer(raw_score_history):
    return raw_score_history / np.abs(np.nanmax(raw_score_history))
