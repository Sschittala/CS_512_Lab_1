import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from q2_gradient import load_train, forward_backward
from q4_sgd import word_error, full_objective

# ---------------------------------------------------------
# 1) Optimized Block Gibbs Sampler
# ---------------------------------------------------------
def mc_sampler(X, W, T, S=10):
    """
    Block Gibbs sampler for linear-chain CRF.
    Returns S samples of label sequences y.
    Complexity: O(S * m * K)
    """
    m, d = X.shape
    K = W.shape[0]
    
    # Pre-calculate emissions to avoid redundant matrix mults
    # This significantly speeds up the sampling loop
    emissions = X @ W.T 

    # Initial y: Independent labels (Step 1 & 2: Ignore edge potential T)
    y = np.zeros(m, dtype=int)
    for s in range(m):
        e_s = emissions[s]
        p = np.exp(e_s - np.max(e_s))
        y[s] = np.random.choice(K, p=p/p.sum())

    samples = []
    for _ in range(S):
        # Step 4: Sample even indices given odd
        for s in range(0, m, 2):
            y[s] = _sample_one_node(s, m, emissions[s], T, y, K)
        # Step 5: Sample odd indices given even
        for s in range(1, m, 2):
            y[s] = _sample_one_node(s, m, emissions[s], T, y, K)
        
        # Step 6: Call current {y} a sample
        samples.append(y.copy())
    return samples

def _sample_one_node(s, m, e_s, T, y, K):
    """Samples one node y_s given its neighbors and emission score."""
    scores = e_s.copy()
    if s > 0:
        scores += T[y[s-1], :] # Potential from y_{s-1} to y_s
    if s < m - 1:
        scores += T[:, y[s+1]] # Potential from y_s to y_{s+1}
    
    # Numerically stable softmax
    p = np.exp(scores - np.max(scores))
    return np.random.choice(K, p=p/np.sum(p))

# ---------------------------------------------------------
# 2) Gradient & Objective with Sampling
# ---------------------------------------------------------
def compute_gradient_sampling(X, y_true, W, T, S=10):
    """Compute gradient using MC samples for marginal approximations."""
    m = len(y_true)
    grad_w = np.zeros_like(W)
    grad_t = np.zeros_like(T)

    # 1. Empirical counts (from the true labels)
    for s in range(m):
        grad_w[y_true[s]] += X[s]
    for s in range(m-1):
        grad_t[y_true[s], y_true[s+1]] += 1

    # 2. Expected counts via MCMC (S samples)
    samples = mc_sampler(X, W, T, S)
    for y_s in samples:
        for s in range(m):
            grad_w[y_s[s]] -= X[s] / S
        for s in range(m-1):
            grad_t[y_s[s], y_s[s+1]] -= 1.0 / S
            
    return grad_w, grad_t

def objective_and_grad_sampling(params, train_x, train_y, d, K, C, S=10):
    """Wrapper for full-batch solvers (LBFGS)."""
    W = params[:K*d].reshape(K, d)
    T = params[K*d:].reshape(K, K)
    
    grad_w_acc = np.zeros_like(W)
    grad_t_acc = np.zeros_like(T)
    total_log_lik = 0
    
    for X, y in zip(train_x, train_y):
        # We use exact DP for the objective scalar to help LBFGS line-search
        log_p, _, _ = forward_backward(X, W, T, y)
        total_log_lik += log_p
        
        # Use sampling for the gradient vector
        gw, gt = compute_gradient_sampling(X, y, W, T, S)
        grad_w_acc += gw
        grad_t_acc += gt

    n = len(train_x)
    obj = -C * (total_log_lik / n) + 0.5 * np.sum(W**2) + 0.5 * np.sum(T**2)
    grad_w = -C * (grad_w_acc / n) + W
    grad_t = -C * (grad_t_acc / n) + T
    
    return obj, np.concatenate([grad_w.ravel(), grad_t.ravel()])

# ---------------------------------------------------------
# 3) Optimization Runners
# ---------------------------------------------------------
def run_sgd(train_x, train_y, test_x, test_y, C, lr, steps, B=32, S=10):
    d, K = 128, 26
    params = np.zeros(K*d + K*K)
    n = len(train_x)
    h_pass, h_obj, h_err = [], [], []

    print("\nRunning SGD (Sampled Gradient)...")
    for step in range(steps):
        indices = np.random.choice(n, B, replace=False)
        gw_batch, gt_batch = np.zeros((K, d)), np.zeros((K, K))
        
        for idx in indices:
            gw, gt = compute_gradient_sampling(train_x[idx], train_y[idx], 
                                               params[:K*d].reshape(K,d), 
                                               params[K*d:].reshape(K,K), S)
            gw_batch += gw
            gt_batch += gt
            
        grad_w = -C * (gw_batch / B) + params[:K*d].reshape(K,d)
        grad_t = -C * (gt_batch / B) + params[K*d:].reshape(K,K)
        params -= lr * np.concatenate([grad_w.ravel(), grad_t.ravel()])

        if step % 5 == 0:
            h_pass.append((step * B) / n)
            h_obj.append(full_objective(params, train_x, train_y, d, K, C))
            h_err.append(word_error(params, test_x, test_y, d, K))
            print(f"Pass {h_pass[-1]:.2f} | Obj: {h_obj[-1]:.2f}")
            
    return params, h_pass, h_obj, h_err

def run_momentum(train_x, train_y, test_x, test_y, C, lr, steps, momentum=0.9, B=32, S=10):
    d, K = 128, 26
    params = np.zeros(K*d + K*K)
    velocity = np.zeros_like(params)
    n = len(train_x)
    h_pass, h_obj, h_err = [], [], []

    print("\nRunning Momentum (Sampled Gradient)...")
    for step in range(steps):
        indices = np.random.choice(n, B, replace=False)
        gw_batch, gt_batch = np.zeros((K, d)), np.zeros((K, K))
        
        for idx in indices:
            gw, gt = compute_gradient_sampling(train_x[idx], train_y[idx], 
                                               params[:K*d].reshape(K,d), 
                                               params[K*d:].reshape(K,K), S)
            gw_batch += gw
            gt_batch += gt
            
        grad = np.concatenate([(-C*(gw_batch/B) + params[:K*d].reshape(K,d)).ravel(),
                               (-C*(gt_batch/B) + params[K*d:].reshape(K,K)).ravel()])
        
        velocity = momentum * velocity - lr * grad
        params += velocity

        if step % 5 == 0:
            h_pass.append((step * B) / n)
            h_obj.append(full_objective(params, train_x, train_y, d, K, C))
            h_err.append(word_error(params, test_x, test_y, d, K))
            print(f"Pass {h_pass[-1]:.2f} | Obj: {h_obj[-1]:.2f}")

    return params, h_pass, h_obj, h_err

def run_lbfgs_sampled(train_x, train_y, test_x, test_y, C, maxiter=20, S=10):
    d, K = 128, 26
    h_pass, h_obj, h_err = [], [], []

    def callback(p):
        h_pass.append(len(h_pass) + 1)
        h_obj.append(full_objective(p, train_x, train_y, d, K, C))
        h_err.append(word_error(p, test_x, test_y, d, K))
        print(f"Iter {len(h_pass)} | Obj: {h_obj[-1]:.2f}")

    print("\nRunning LBFGS (Sampled Gradient)...")
    res = fmin_l_bfgs_b(objective_and_grad_sampling, np.zeros(K*d + K*K), 
                        args=(train_x, train_y, d, K, C, S),
                        callback=callback, maxiter=maxiter, pgtol=1e-3, disp=1)
    return res[0], h_pass, h_obj, h_err

def plot_results(sgd_hist, mom_hist, lbfgs_hist):
    # Extract data from histories
    # (params, history_pass, history_obj, history_err)
    _, sgd_pass, sgd_obj, sgd_err = sgd_hist
    _, mom_pass, mom_obj, mom_err = mom_hist
    _, lbfgs_pass, lbfgs_obj, lbfgs_err = lbfgs_hist

    # --- Plot 1: Training Objective ---
    plt.figure(figsize=(8, 6))
    plt.plot(sgd_pass, sgd_obj, label='SGD (S=10)', color='tab:blue', linewidth=1.5)
    plt.plot(mom_pass, mom_obj, label='Momentum (S=10)', color='tab:orange', linewidth=1.5)
    plt.plot(lbfgs_pass, lbfgs_obj, label='LBFGS (S=10)', color='tab:green', marker='o', markersize=4)
    
    plt.xlabel('Effective Passes (Epochs)')
    plt.ylabel('Negative Log-Likelihood Objective')
    plt.title('Training Objective: MCMC Sampling Gradient')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../result/q4b_objective.png', dpi=300)
    plt.show()

    # --- Plot 2: Test Word Error ---
    plt.figure(figsize=(8, 6))
    plt.plot(sgd_pass, sgd_err, label='SGD (S=10)', color='tab:blue', linewidth=1.5)
    plt.plot(mom_pass, mom_err, label='Momentum (S=10)', color='tab:orange', linewidth=1.5)
    plt.plot(lbfgs_pass, lbfgs_err, label='LBFGS (S=10)', color='tab:green', marker='o', markersize=4)
    
    plt.xlabel('Effective Passes (Epochs)')
    plt.ylabel('Word Error Rate')
    plt.title('Test Word Error: MCMC Sampling Gradient')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('../result/q4b_word_error.png', dpi=300)
    plt.show()
# ---------------------------------------------------------
# 4) Main Execution & Plotting
# ---------------------------------------------------------
if __name__ == "__main__":
    train_x, train_y = load_train("../data/train.txt")
    test_x, test_y = load_train("../data/test.txt")
    C = 1000
    S = 10 

    # Recommended hyperparameters from standard SGD tuning
    sgd_res = run_sgd(train_x, train_y, test_x, test_y, C, lr=1e-4, steps=2000, S=S)
    mom_res = run_momentum(train_x, train_y, test_x, test_y, C, lr=1e-4, steps=2000, S=S)
    lbfgs_res = run_lbfgs_sampled(train_x, train_y, test_x, test_y, C, maxiter=25, S=S)
    plot_results(sgd_res, mom_res, lbfgs_res)