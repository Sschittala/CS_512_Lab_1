import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from q2_gradient import load_train
from q4_sgd import word_error, full_objective

# -----------------------------
# 1) Block Gibbs Sampler
# -----------------------------
def mc_sampler(X, W, T, S=10):
    """
    Block Gibbs sampler for linear-chain CRF.
    Returns S samples of label sequences y.
    """
    m, d = X.shape
    K = W.shape[0]

    # Initialize y from independent distribution ignoring edges
    y = np.zeros(m, dtype=int)
    for s in range(m):
        scores = W @ X[s]
        scores -= np.max(scores)
        probs = np.exp(scores)
        probs /= np.sum(probs)
        y[s] = np.random.choice(K, p=probs)

    samples = []
    for _ in range(S):
        # Sample odd positions conditioned on even
        for s in range(1, m, 2):
            scores = W @ X[s]
            if s > 0:
                scores += T[y[s-1], :]
            if s < m-1:
                scores += T[:, y[s+1]]
            scores -= np.max(scores)
            probs = np.exp(scores)
            probs /= np.sum(probs)
            y[s] = np.random.choice(K, p=probs)
        # Sample even positions conditioned on odd
        for s in range(0, m, 2):
            scores = W @ X[s]
            if s > 0:
                scores += T[y[s-1], :]
            if s < m-1:
                scores += T[:, y[s+1]]
            scores -= np.max(scores)
            probs = np.exp(scores)
            probs /= np.sum(probs)
            y[s] = np.random.choice(K, p=probs)

        samples.append(y.copy())
    return samples

# -----------------------------
# 2) Compute Monte Carlo gradient
# -----------------------------
def compute_gradient_sampling(X, y_true, W, T, S=10):
    """
    Compute gradient using MC samples.
    Returns grad_w, grad_t, and log-prob estimate for true y.
    """
    m = len(y_true)
    K = W.shape[0]

    grad_w = np.zeros_like(W)
    grad_t = np.zeros_like(T)

    # Empirical counts from true label
    for s in range(m):
        grad_w[y_true[s]] += X[s]
    for s in range(m-1):
        grad_t[y_true[s], y_true[s+1]] += 1

    # Draw S samples
    samples = mc_sampler(X, W, T, S)

    # Approximate expectation under model
    for y in samples:
        for s in range(m):
            grad_w[y[s]] -= X[s] / S
        for s in range(m-1):
            grad_t[y[s], y[s+1]] -= 1.0 / S

    # Optional: approximate log probability of true y
    log_prob = 0  # we won’t compute it exactly for efficiency

    return grad_w, grad_t, log_prob

# -----------------------------
# 3) Objective and gradient
# -----------------------------
def objective_and_grad_sampling(params, train_x, train_y, d=128, K=26, C=1000, S=10):
    """
    Return objective and gradient for all training data using MC sampling
    """
    W = params[:K*d].reshape(K, d)
    T = params[K*d:].reshape(K, K)

    grad_w = np.zeros_like(W)
    grad_t = np.zeros_like(T)
    total_log = 0.0

    for X, y in zip(train_x, train_y):
        gw, gt, log_p = compute_gradient_sampling(X, y, W, T, S)
        grad_w += gw
        grad_t += gt
        total_log += log_p  # optional, 0 here
        
    n = len(train_x)
    avg_log = total_log / n
    avg_grad_w = grad_w / n
    avg_grad_t = grad_t / n
    
    # Regularization
    obj = -C * avg_log + 0.5 * np.sum(W**2) + 0.5 * np.sum(T**2)
    grad_w = -C * avg_grad_w + W
    grad_t = -C * avg_grad_t + T
    grad = np.concatenate([grad_w.ravel(), grad_t.ravel()])

    return obj, grad

# -----------------------------
# 4) Mini-batch gradient for SGD/Momentum
# -----------------------------
def minibatch_obj_and_grad(params, words_x, words_y, batch_idx, d=128, K=26, C=1000, S=10):
    W = params[:K*d].reshape(K, d)
    T = params[K*d:].reshape(K, K)

    grad_w_sum = np.zeros_like(W)
    grad_t_sum = np.zeros_like(T)

    for i in batch_idx:
        X = words_x[i]
        y = words_y[i]
        gw, gt, _ = compute_gradient_sampling(X, y, W, T, S)
        grad_w_sum += gw
        grad_t_sum += gt

    B = len(batch_idx)
    avg_grad_w = grad_w_sum / B
    avg_grad_t = grad_t_sum / B

    obj = 0.5 * np.sum(W**2) + 0.5 * np.sum(T**2)
    grad_w = -C * avg_grad_w + W
    grad_t = -C * avg_grad_t + T
    grad = np.concatenate([grad_w.ravel(), grad_t.ravel()])

    return obj, grad

# -----------------------------
# 5) SGD
# -----------------------------
def run_sgd(train_x, train_y, test_x, test_y, sample_rate, C=1000, B=32, lr=1e-3, steps=2000, seed=0, S=10):
    rng = np.random.default_rng(seed)
    d = 128
    K = 26
    params = np.zeros(K*d + K*K)

    history_pass, history_obj, history_err = [], [], []
    n = len(train_x)

    print(f"\nStarting SGD training (lr={lr}, batch={B}, steps={steps})")
    for step in range(steps):
        batch_idx = rng.choice(n, size=B, replace=False)
        _, grad = minibatch_obj_and_grad(params, train_x, train_y, batch_idx, d, K, C, S)
        params -= lr * grad

        eff_pass = (step + 1) * B / n
        if step % sample_rate == 0:
            history_pass.append(eff_pass)
            history_obj.append(full_objective(params, train_x, train_y, d, K, C))
            history_err.append(word_error(params, test_x, test_y, d, K))
            print(f"[SGD] step={step:4} pass={eff_pass:.3f} obj={history_obj[-1]:.4f} test_err={history_err[-1]:.4f}")

    return params, history_pass, history_obj, history_err

# -----------------------------
# 6) SGD with Momentum
# -----------------------------
def run_sgd_momentum(train_x, train_y, test_x, test_y, sample_rate, C=1000, B=32, lr=1e-3, steps=2000, momentum=0.9, seed=0, S=10):
    rng = np.random.default_rng(seed)
    d = 128
    K = 26
    params = np.zeros(K*d + K*K)
    velocity = np.zeros_like(params)

    history_pass, history_obj, history_err = [], [], []
    n = len(train_x)

    print(f"\nStarting Momentum training (lr={lr}, batch={B}, momentum={momentum})")
    for step in range(steps):
        batch_idx = rng.choice(n, size=B, replace=False)
        _, grad = minibatch_obj_and_grad(params, train_x, train_y, batch_idx, d, K, C, S)

        velocity = momentum * velocity - lr * grad
        params += velocity

        eff_pass = (step + 1) * B / n
        if step % sample_rate == 0:
            history_pass.append(eff_pass)
            history_obj.append(full_objective(params, train_x, train_y, d, K, C))
            history_err.append(word_error(params, test_x, test_y, d, K))
            print(f"[Momentum] step={step:4} pass={eff_pass:.3f} obj={history_obj[-1]:.4f} test_err={history_err[-1]:.4f}")

    return params, history_pass, history_obj, history_err

# -----------------------------
# 7) LBFGS
# -----------------------------
def run_lbfgs(train_x, train_y, test_x, test_y, C=1000, maxfun=300, S=10):
    d = 128
    K = 26
    params = np.zeros(K*d + K*K)
    history_pass, history_obj, history_err = [], [], []
    eval_count = {"n": 0}

    def tracked_obj(p, *args):
        eval_count["n"] += 1
        return objective_and_grad_sampling(p, *args)

    def callback(p):
        history_pass.append(eval_count["n"])
        history_obj.append(full_objective(p, train_x, train_y, d, K, C))
        history_err.append(word_error(p, test_x, test_y, d, K))
        print(f"[LBFGS] step={len(history_pass):4} obj={history_obj[-1]:.4f} test_err={history_err[-1]:.4f}")

    params_opt, nfeval, rc = fmin_tnc(
        func=lambda p, *args: tracked_obj(p, *args),
        x0=params,
        args=(train_x, train_y, d, K, C, S),
        callback=callback,
        maxfun=maxfun,
        messages=5
    )
    return params_opt, history_pass, history_obj, history_err

# -----------------------------
# 8) Plotting histories
# -----------------------------
def plot_histories(sgd_hist, mom_hist, lbfgs_hist):
    _, sgd_pass, sgd_obj, sgd_err = sgd_hist
    _, mom_pass, mom_obj, mom_err = mom_hist
    _, lbfgs_eval, lbfgs_obj, lbfgs_err = lbfgs_hist

    plt.figure(figsize=(7,5))
    plt.plot(sgd_pass, sgd_obj, label="SGD")
    plt.plot(mom_pass, mom_obj, label="Momentum")
    plt.plot(lbfgs_eval, lbfgs_obj, label="LBFGS")

    plt.xlabel("Effective passes")
    plt.ylabel("Training objective")
    plt.title("Training Objective")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("result/q4b_objective_SGD_momentum.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(sgd_pass, sgd_err, label="SGD")
    plt.plot(mom_pass, mom_err, label="Momentum")
    plt.plot(lbfgs_eval, lbfgs_err, label="LBFGS")

    plt.xlabel("Effective passes")
    plt.ylabel("Test word error")
    plt.title("Test Word Error")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("result/q4b_word_error.png", dpi=200)
    plt.close()

# -----------------------------
# 9) Run everything
# -----------------------------
train_x, train_y = load_train("data/train.txt")
test_x, test_y = load_train("data/test.txt")

C = 1000
S = 10

sgd_hist = run_sgd(train_x, train_y, test_x, test_y, sample_rate=5, C=C, B=32, lr=1e-4, steps=20, S=S)
mom_hist = run_sgd_momentum(train_x, train_y, test_x, test_y, sample_rate=5, C=C, B=32, lr=1e-4, momentum=0.9, steps=20, S=S)
lbfgs_hist = run_lbfgs(train_x, train_y, test_x, test_y, C=C, maxfun=20, S=S)
plot_histories(sgd_hist, mom_hist, lbfgs_hist)