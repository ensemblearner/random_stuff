"""
http://www.cs.rpi.edu/~drinep/Papers/Drineas_SODA_06.pdf
"""

import numpy as np

def vec_norm(vec):
    return np.linalg.norm(vec)

def compute_probabilities(A, B):
    p_i = map(lambda x: vec_norm(x), A)
    p_j = map(lambda x: vec_norm(x), B.T)
    probs = map(lambda x,y: x*y, p_i, p_j)
    return probs/(sum(p_i)*sum(p_j))

A = np.random.randn(20, 2)
B = np.random.randn(2, 20)
sampling_probs = compute_probabilities(A, B)
