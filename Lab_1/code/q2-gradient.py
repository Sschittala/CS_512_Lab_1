import numpy as np
from scipy.optimize import fmin_tnc

def logsumexp(arr, axis=None):
    max_val = np.max(arr, axis=axis, keepdims=True)
    return np.squeeze(max_val + np.log(np.sum(np.exp(arr - max_val), axis=axis, keepdims=True)))

def forward_backward(X, W, T, y_true):
    m, d = X.shape
    K = W.shape[0]
    node_score = X @ W.T
    alpha = np.zeros((m, K))
    alpha[0] = node_score[0]
    for s in range(1, m):
        scores = alpha[s-1][:, None] + T
        alpha[s] = node_score[s] + logsumexp(scores, axis=0)
    beta = np.zeros((m, K))
    for s in reversed(range(m-1)):
        scores = T + node_score[s+1][None, :] + beta[s+1][None, :]
        beta[s] = logsumexp(scores, axis=1)
    logZ = logsumexp(alpha[-1])
    marg_node = np.exp(alpha + beta - logZ)
    marg_edge = np.exp(alpha[:-1][:, :, None] + T[None, :, :] + node_score[1:][:, None, :] + beta[1:][:, None, :] - logZ)
    log_p = np.sum(node_score[np.arange(m), y_true]) + np.sum(T[y_true[:-1], y_true[1:]]) - logZ
    return log_p, marg_node, marg_edge

def compute_gradient(X, y_true, W, T):
    log_p, marg_node, marg_edge = forward_backward(X, W, T, y_true)
    indicator = np.zeros_like(marg_node)
    indicator[np.arange(X.shape[0]), y_true] = 1
    grad_w = (indicator - marg_node).T @ X
    m, K = marg_node.shape
    indicator_edge = np.zeros((m-1, K, K))
    indicator_edge[np.arange(m-1), y_true[:-1], y_true[1:]] = 1
    grad_t = np.sum(indicator_edge - marg_edge, axis=0)
    return log_p, grad_w, grad_t

def load_model(filename):
    vals = np.loadtxt(filename)
    d = 128
    K = 26
    W = vals[:K*d].reshape(K, d)
    T = vals[K*d:].reshape(K, K)
    return W, T

def load_train(filename):
    words_x = []
    words_y = []
    data = np.loadtxt(filename, dtype=str)
    word_ids = np.unique(data[:, 0])
    for wid in word_ids:
        word_data = data[data[:, 0] == wid]
        X = word_data[:, 5:].astype(float)
        y = np.array([ord(ch) - ord('a') for ch in word_data[:, 1]])
        words_x.append(X)
        words_y.append(y)
    return words_x, words_y

def compute_full_gradient(words_x, words_y, W, T):
    total_log = 0.0
    grad_w_sum = np.zeros_like(W)
    grad_t_sum = np.zeros_like(T)
    n = len(words_x)
    for X, y in zip(words_x, words_y):
        log_p, grad_w, grad_t = compute_gradient(X, y, W, T)
        total_log += log_p
        grad_w_sum += grad_w
        grad_t_sum += grad_t
    avg_loop = total_log / n
    avg_grad_w = grad_w_sum / n
    avg_grad_t = grad_t_sum / n
    return avg_loop, avg_grad_w, avg_grad_t
def objective_and_grad(params, words_x, words_y, d=128, K=26, C=1000):
    W = params[:K*d].reshape(K, d)
    T = params[K*d:].reshape(K, K)
    total_log = 0.0
    grad_w_sum = np.zeros_like(W)
    grad_t_sum = np.zeros_like(T)
    for X, y in zip(words_x, words_y):
        log_p, grad_w, grad_t = compute_gradient(X, y, W, T)
        total_log += log_p
        grad_w_sum += grad_w
        grad_t_sum += grad_t
    n = len(words_x)
    avg_grad_w = grad_w_sum / n
    avg_grad_t = grad_t_sum / n
    obj = -C * total_log + 0.5 * np.sum(W**2) + 0.5 * np.sum(T**2)
    grad_w = -C * avg_grad_w + W
    grad_t = -C * avg_grad_t + T
    grad = np.concatenate([grad_w.flatten(), grad_t.flatten('F')])
    return obj, grad
    
def save_solution(filename, grad_w, grad_t):
    vec = np.concatenate([grad_w.flatten(), grad_t.flatten('F')])
    np.savetxt(filename, vec)

# #2a
# W, T = load_model("/Users/tylerstrach/Desktop/10 - Spring 2026/lab1/CS_512_Lab_1/Lab_1/data/model.txt")
# words_x, words_y = load_train("/Users/tylerstrach/Desktop/10 - Spring 2026/lab1/CS_512_Lab_1/Lab_1/data/train.txt")
# avg_log, avg_grad_w, avg_grad_t = compute_full_gradient(words_x, words_y, W, T) 
# save_solution("../result/gradient.txt", avg_grad_w, avg_grad_t)
# print("Average log p(y|X) over training set:", avg_log) # -4.140274439213334

# #2b
# W_init = np.zeros((26, 128))
# T_init = np.zeros((26, 26))
# params_init = np.concatenate([W_init.flatten(), T_init.flatten('F')])
# solution = fmin_tnc(func=lambda p, *args: objective_and_grad(p, *args), x0=params_init, args=(words_x, words_y), messages=0)
# params_opt = solution[0]
# W_opt = params_opt[:26*128].reshape(26, 128)
# T_opt = params_opt[26*128:].reshape(26, 26, order='F')
# save_solution("../result/solution.txt", W_opt, T_opt)


 
  