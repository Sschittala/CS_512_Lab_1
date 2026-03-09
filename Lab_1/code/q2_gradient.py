import numpy as np
from scipy.optimize import fmin_tnc
from collections import defaultdict

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
    """
    Loads data from the CRF letter-level dataset and groups letters by word.

    Returns:
        words_x: list of np.arrays, each array = (word_length, 128) features
        words_y: list of np.arrays, each array = (word_length,) labels 0..25
    """

    data = np.loadtxt(filename, dtype=str)
    groups = defaultdict(list)
    word_order = []

    for row in data:
        letter = row[1]  # column 1 = letter
        word_id = int(row[3]) # column 3 = word_id
        pos = int(row[4])     # column 4 = letter position in word
        features = row[5:].astype(float)
        
        if word_id not in groups:
            word_order.append(word_id)
            
        groups[word_id].append((pos, features, ord(letter) - ord('a')))

    words_x, words_y = [], []
    # if we encounter a new word, store the previous one
    for wid in word_order:
        items = sorted(groups[wid], key=lambda t: t[0])
        X = np.array([feat for _, feat, _ in items], dtype=float)
        y = np.array([lab for _, _, lab in items], dtype=int)
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
    avg_log = total_log / n
    avg_grad_w = grad_w_sum / n
    avg_grad_t = grad_t_sum / n
    
    obj = -C * avg_log + 0.5 * np.sum(W**2) + 0.5 * np.sum(T**2)
    grad_w = -C * avg_grad_w + W
    grad_t = -C * avg_grad_t + T
    grad = np.concatenate([grad_w.ravel(), grad_t.ravel()])
    return obj, grad
    
def save_solution(filename, grad_w, grad_t):
    vec = np.concatenate([grad_w.ravel(), grad_t.ravel()])
    np.savetxt(filename, vec)

# #2a
'''
W, T = load_model("../data/model.txt")
words_x, words_y = load_train("../data/train.txt")
avg_log, avg_grad_w, avg_grad_t = compute_full_gradient(words_x, words_y, W, T) 
save_solution("../result/gradient.txt", avg_grad_w, avg_grad_t)
print("Average log p(y|X) over training set:", avg_log) # -4.140274439213334

# #2b
W_init = np.zeros((26, 128))
T_init = np.zeros((26, 26))
params_init = np.concatenate([W_init.ravel(), T_init.ravel()])
solution = fmin_tnc(
        func=lambda p, *args: objective_and_grad(p, *args),
        x0=params_init,
        args=(words_x, words_y),
        maxfun=100,
        ftol=1e-3,
        messages=5
)

params_opt = solution[0]
W_opt = params_opt[:26*128].reshape(26, 128)
T_opt = params_opt[26*128:].reshape(26, 26)
save_solution("../result/solution.txt", W_opt, T_opt)
'''


 
  
