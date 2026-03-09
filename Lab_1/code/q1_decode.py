import numpy as np
from itertools import product

def decode_dp(X, W, T):
    m = len(X)
    K = 26
    delta = np.zeros((m, K))
    psi = np.zeros((m, K), dtype=int)
    for j in range(K):
        delta[0, j] = np.dot(W[j], X[0])
    for s in range(1, m):
        for j in range(K):
            scores = delta[s-1] + T[:, j]
            psi[s, j] = np.argmax(scores)
            delta[s, j] = np.dot(W[j], X[s]) + np.max(scores)
    y = np.zeros(m, dtype=int)
    y[m-1] = np.argmax(delta[m-1])
    for s in reversed(range(m-1)):
        y[s] = psi[s+1, y[s+1]]
    return y + 1

def decode_bruteforce(X, W, T):
    m = len(X)
    K = 26
    best_score = -np.inf
    best_y = None
    for y in product(range(K), repeat=m):
        score = 0.0
        for s in range(m):
            score += np.dot(W[y[s]], X[s])
        for s in range(m - 1):
            score += T[y[s], y[s+1]]
        if score > best_score:
            best_score = score
            best_y = y
    return np.array(best_y) + 1, best_score

# data = np.loadtxt("/Users/sai/CS_512_Lab_1-2/Lab_1/data/decode_input.txt")
# m = 100
# d = 128
# K = 26
# X = data[:m*d].reshape(m, d)
# W = data[m*d : m*d + K*d].reshape(K, d)
# T = data[m*d + K*d :].reshape(K, K)

# y_full = decode_dp(X, W, T)
# np.savetxt("result/decode_output.txt", y_full, fmt='%d')

# score_full = 0
# m = len(X)
# for s in range(m):
#     score_full += np.dot(W[y_full[s]-1], X[s])
# for s in range(m-1):
#     score_full += T[y_full[s]-1, y_full[s+1]-1]
# print("Maximum objective value:", score_full)
# ## Maximum objective value: 200.18515048829283