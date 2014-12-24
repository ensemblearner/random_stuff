"""
http://www.cs.rpi.edu/~drinep/Papers/Drineas_SODA_06.pdf
"""

import numpy as np

def vec_norm(vec):
    return np.linalg.norm(vec)

def compute_probabilities(A, B):
    """
    P_i = || A~i|| * ||B_j||/sum(|| A~i|| * ||B_j||)
    """
    p_i = map(lambda x: vec_norm(x), A.T)
    p_j = map(lambda x: vec_norm(x), B)
    probs = map(lambda x,y: x*y, p_i, p_j)
    return probs/(sum(p_i)*sum(p_j))

def approximate_matrices(A, B, probs, n, c):
    S = np.zeros((n, c))
    D = np.zeros((c, c))
    normed_probs = probs/sum(probs)
    for t in range(c):
        prob = np.random.choice(probs, p = normed_probs)
        index = np.where(probs==prob)[0][0]
        D[t][t] = 1/np.sqrt(c*prob)
        S[index][t] = 1

    C = np.dot(np.dot(A, S), D)
    R = np.dot(np.dot(D, S.T), B)
    return C, R
m = 20
n = 4
d = 20
c = 2
A = np.random.randn(m, n)
B = np.random.randn(n, d)
sampling_probs = compute_probabilities(A, B)

C, R = approximate_matrices(A, B, sampling_probs, n, c)
approx_prod = np.dot(C, R)
true_prod = np.dot(A, B)

