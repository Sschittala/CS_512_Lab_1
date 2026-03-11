import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc
from q2_gradient import load_train
from q4_sgd import  word_error, full_objective

def mc_sampler(X, W, T, S=5):
    m, d = X.shape
    K = W.shape[0]
    y = np.zeros(m, dtype=int)
    for s in range(m):
        scores = W @ X[s]
        scores -= np.max(scores)
        probs = np.exp(scores)
        probs /= np.sum(probs)
        y[s] = np.random.choice(K, p=probs)
    samples = []
    for _ in range(S):
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

def compute_gradient_sampling(X, y_true, W, T, S=5):
    m = len(y_true)
    K = W.shape[0]
    grad_w = np.zeros_like(W)
    grad_t = np.zeros_like(T)
    for s in range(m):
        grad_w[y_true[s]] += X[s]
    for s in range(m-1):
        grad_t[y_true[s], y_true[s+1]] += 1
    samples = mc_sampler(X, W, T, S)
    for y in samples:
        for s in range(m):
            grad_w[y[s]] -= X[s] / S
        for s in range(m-1):
            grad_t[y[s], y[s+1]] -= 1.0 / S
    log_p = 0
    return log_p, grad_w, grad_t

def minibatch_obj_and_grad(params, words_x, words_y, batch_idx, d=128, K=26, C=1000, S=5):
    W = params[:K*d].reshape(K, d)
    T = params[K*d:].reshape(K, K)

    total_log = 0.0
    grad_w_sum = np.zeros_like(W)
    grad_t_sum = np.zeros_like(T)

    for i in batch_idx:
        X = words_x[i]
        y = words_y[i]
        log_p, grad_w, grad_t = compute_gradient_sampling(X, y, W, T, S)
        total_log += log_p
        grad_w_sum += grad_w
        grad_t_sum += grad_t

    B = len(batch_idx)
    avg_log = total_log / B
    avg_grad_w = grad_w_sum / B
    avg_grad_t = grad_t_sum / B

    obj = -C * avg_log + 0.5 * np.sum(W**2) + 0.5 * np.sum(T**2)
    grad_w = -C * avg_grad_w + W
    grad_t = -C * avg_grad_t + T
    grad = np.concatenate([grad_w.ravel(), grad_t.ravel()])

    return obj, grad

def objective_and_grad_sampling(params, train_x, train_y, d=128, K=26, C=1000, S=5):

    W = params[:K*d].reshape(K, d)
    T = params[K*d:].reshape(K, K)

    total_log = 0
    grad_w = np.zeros_like(W)
    grad_t = np.zeros_like(T)

    for X, y in zip(train_x, train_y):
        log_p, gw, gt = compute_gradient_sampling(X, y, W, T, S)
        total_log += log_p
        grad_w += gw
        grad_t += gt

    obj = -C * total_log + 0.5 * np.sum(W**2) + 0.5 * np.sum(T**2)

    grad_w = -C * grad_w + W
    grad_t = -C * grad_t + T

    grad = np.concatenate([grad_w.ravel(), grad_t.ravel()])

    return obj, grad

def run_sgd(train_x, train_y, test_x, test_y, sample_rate, C=1000, B=32, lr=1e-3, steps=2000, seed=0, S=5):
    rng = np.random.default_rng(seed)
    d = 128
    K = 26
    params = np.zeros(K*d + K*K)

    history_pass = []
    history_obj = []
    history_err = []

    n = len(train_x)

    # Main training loop. Run for 'steps' steps
    print(f"\nStarting SGD training (lr={lr}, batch={B}, steps={steps})")
    for step in range(steps):
        
        batch_idx = rng.choice(n, size=B, replace=False)
        _, grad = minibatch_obj_and_grad(params, train_x, train_y, batch_idx, d, K, C, S)
        params = params - lr * grad

        eff_pass = (step + 1) * B / n
       
        if step % sample_rate == 0:
            history_pass.append(eff_pass)
            history_obj.append(full_objective(params, train_x, train_y, d, K, C))
            history_err.append(word_error(params, test_x, test_y, d, K))
            print(
                f"  [SGD] step={step:4} "
                f"pass={eff_pass:.3f} "
                f"obj={history_obj[-1]:.4f} "
                f"test_err={history_err[-1]:.4f}"
            )

    print(f"Final training objective: {history_obj[-1]:.4f}")
    print(f"Final test word error: {history_err[-1]:.4f}")

    return params, history_pass, history_obj, history_err

def run_sgd_momentum(train_x, train_y, test_x, test_y, sample_rate, C=1000, B=32, lr=1e-3, steps=2000,
                     momentum=0.9, seed=0, S=5):
    rng = np.random.default_rng(seed)
    d = 128
    K = 26
    params = np.zeros(K*d + K*K)
    velocity = np.zeros_like(params)

    history_pass = []
    history_obj = []
    history_err = []

    n = len(train_x)

    # Main training loop. Run for 'steps' steps with momentum
    print(f"\nStarting Momentum training (lr={lr}, batch={B}, momentum={momentum})")
    for step in range(steps):
        
        batch_idx = rng.choice(n, size=B, replace=False)
        _, grad = minibatch_obj_and_grad(params, train_x, train_y, batch_idx, d, K, C, S)

        # Add momentum to 'params'
        velocity = momentum * velocity - lr * grad
        params = params + velocity

        eff_pass = (step + 1) * B / n

        if step % sample_rate == 0:
            history_pass.append(eff_pass)
            history_obj.append(full_objective(params, train_x, train_y, d, K, C))
            history_err.append(word_error(params, test_x, test_y, d, K))

            print(
                f"  [Momentum] step={step:4} "
                f"pass={eff_pass:.3f} "
                f"obj={history_obj[-1]:.4f} "
                f"test_err={history_err[-1]:.4f}"
            )


    print(f"Final training objective: {history_obj[-1]:.4f}")
    print(f"Final test word error: {history_err[-1]:.4f}")

    return params, history_pass, history_obj, history_err

def run_lbfgs(train_x, train_y, test_x, test_y, C=1000, maxfun=300, S=5):
    d = 128
    K = 26
    params = np.zeros(K*d + K*K)

    eval_count = {"n": 0}
    history_pass = []
    history_obj = []
    history_err = []

    def tracked_obj(p, *args):
        eval_count["n"] += 1
        return objective_and_grad_sampling(p, *args)

    def callback(p):
        history_pass.append(eval_count["n"])
        history_obj.append(full_objective(p, train_x, train_y, d, K, C))
        history_err.append(word_error(p, test_x, test_y, d, K))
        print(
            f"  [LBFGS] step={len(history_pass):4} "
            f"obj={history_obj[-1]:.4f} "
            f"test_err={history_err[-1]:.4f}"
        )


    params_opt, nfeval, rc = fmin_tnc(
        func=lambda p, *args: tracked_obj(p, *args),
        x0=params,
        args=(train_x, train_y, d, K, C, S),
        callback=callback,
        maxfun=maxfun,
        messages=5
    )

    return params_opt, history_pass, history_obj, history_err

def plot_histories(sgd_hist, mom_hist, lbfgs_hist):
    _, sgd_pass, sgd_obj, sgd_err = sgd_hist
    _, mom_pass, mom_obj, mom_err = mom_hist
    _, lbfgs_eval, lbfgs_obj, lbfgs_err = lbfgs_hist

    # 1) SGD + Momentum: training objective vs effective passes
    plt.figure(figsize=(7, 5))
    plt.plot(sgd_pass, sgd_obj, label="SGD")
    plt.plot(mom_pass, mom_obj, label="Momentum")
    plt.xlabel("Effective number of passes")
    plt.ylabel("Training objective")
    plt.title("Q4b: Sampling gradient")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("../result/q4b_objective_SGD_momentum.png", dpi=200)
    plt.close()

    # 2) LBFGS: training objective vs function evaluations
    # plt.figure(figsize=(7, 5))
    # plt.plot(lbfgs_eval, lbfgs_obj, label="LBFGS")
    # plt.xlabel("Number of function evaluations")
    # plt.ylabel("Training objective")
    # plt.title("Q4b: Sampling gradient")
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.tight_layout()
    # plt.savefig("../result/q4b_objective_lbfgs.png", dpi=200)
    # plt.close()

    # 3) SGD + Momentum: test word-wise error vs effective passes
    plt.figure(figsize=(7, 5))
    plt.plot(sgd_pass, sgd_err, label="SGD")
    plt.plot(mom_pass, mom_err, label="Momentum")
    plt.xlabel("Effective number of passes")
    plt.ylabel("Test word-wise error")
    plt.title("Q4b: Test word-wise error (Sampling gradient)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("../result/q4b_word_error_sgd_momentum.png", dpi=200)
    plt.close()

    # 4) LBFGS: test word-wise error vs function evaluations
    # plt.figure(figsize=(7, 5))
    # plt.plot(lbfgs_eval, lbfgs_err, label="LBFGS")
    # plt.xlabel("Number of function evaluations")
    # plt.ylabel("Test word-wise error")
    # plt.title("Q4b: Test word-wise error (LBFGS)")
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.6)
    # plt.tight_layout()
    # plt.savefig("../result/q4b_word_error_lbfgs.png", dpi=200)
    # plt.close()

train_x, train_y = load_train("../data/train.txt")
test_x, test_y = load_train("../data/test.txt")

C = 10000 # Choose optimal C
S = 5

sgd_hist = run_sgd(train_x, train_y, test_x, test_y, 5, C=C, B=32, lr=1e-4, steps=200, S=S)
mom_hist = run_sgd_momentum(train_x, train_y, test_x, test_y, 5, C=C, B=32, lr=1e-4, momentum=0.9, steps=200, S=S)
lbfgs_hist = run_lbfgs(train_x, train_y, test_x, test_y, C=C, maxfun=200, S=S)
plot_histories(sgd_hist, mom_hist, lbfgs_hist)

             