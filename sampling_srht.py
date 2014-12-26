"""
http://arxiv.org/pdf/0710.1435.pdf
"""
import numpy as np

from scipy.linalg import hadamard

def sampling_matrix(n, r):
    sample_indices = np.random.choice(n, r)
    S = np.zeros((n, r))
    scaling_factor = np.sqrt(n/r)
    for i, index in enumerate(sample_indices):
        S[index][i] = scaling_factor
    return S

def hadamard_transform(n):
    H = hadamard(n)
    d = np.random.choice([-1, 1], size=n)
    D = np.zeros((n, n))
    np.fill_diagonal(D, d)
    return H, D

def find_optimum(A, b, r):
    n, d = A.shape
    S = sampling_matrix(n, r)
    H, D = hadamard_transform(n)
    x_opt = np.linalg.pinv(S.T.dot(H).dot(D).dot(A)).dot(S.T).dot(H).dot(D).dot(b)
    return x_opt

if __name__ == '__main__':
    n,d = 1024, 25
    r = 10
    A = np.random.randn(n, d)
    b = np.random.randn(n)
    true_x = np.linalg.pinv(A).dot(b)
    x_opt = find_optimum(A, b, r)
    print "approximation error ", np.linalg.norm(true_x-x_opt)
