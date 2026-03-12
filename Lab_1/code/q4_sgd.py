# This file implements a stochastic gradient descent sampling algorithm
# for learning
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

from q1_decode import decode_dp
from q2_gradient import load_train, objective_and_grad, compute_gradient



# Utility function for SGD training, takes 'batch_idx' as additional input specifying
# batching composition of training data
def minibatch_obj_and_grad(params, words_x, words_y, batch_idx, d=128, K=26, C=1000):
    W = params[:K*d].reshape(K, d)
    T = params[K*d:].reshape(K, K)

    total_log = 0.0
    grad_w_sum = np.zeros_like(W)
    grad_t_sum = np.zeros_like(T)

    for i in batch_idx:
        X = words_x[i]
        y = words_y[i]
        log_p, grad_w, grad_t = compute_gradient(X, y, W, T)
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

# Utility function for calculating objective
def full_objective(params, words_x, words_y, d=128, K=26, C=1000):
    obj, _ = objective_and_grad(params, words_x, words_y, d=d, K=K, C=C)
    return obj

# Run model prediction on words given set of parameters
def predict_words(params, test_x, d=128, K=26):
    W = params[:K*d].reshape(K, d)
    T = params[K*d:].reshape(K, K)
    preds = []
    for X in test_x:
        preds.append(decode_dp(X, W, T) - 1) # compare against 0..25 labels
    return preds

# Utility function for obtaining accuracy of word predictions
def word_accuracy(y_true_seq, y_pred_seq):
    total_words = len(y_true_seq)
    correct_words = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        if np.array_equal(y_true, y_pred):
            correct_words += 1
    return correct_words / total_words

# Utility function for determining word error
def word_error(params, test_x, test_y, d=128, K=26):
    preds = predict_words(params, test_x, d=d, K=K)
    acc = word_accuracy(test_y, preds)
    return 1.0 - acc

# Implement plain SGD without momentum
def run_sgd(train_x, train_y, test_x, test_y, sample_rate, C=1000, B=32, lr=1e-3, steps=2000, seed=0):
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
        _, grad = minibatch_obj_and_grad(params, train_x, train_y, batch_idx, d, K, C)
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

# Implement SGD with momentum
def run_sgd_momentum(train_x, train_y, test_x, test_y, sample_rate, C=1000, B=32, lr=1e-3, steps=2000,
                     momentum=0.9, seed=0):
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
        _, grad = minibatch_obj_and_grad(params, train_x, train_y, batch_idx, d, K, C)

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


def run_lbfgs(train_x, train_y, test_x, test_y, C=1000, maxfun=100):
    d = 128
    K = 26
    params = np.zeros(K*d + K*K)

    eval_count = {"n": 0}
    history_pass = []
    history_obj = []
    history_err = []

    def tracked_obj(p, *args):
        return objective_and_grad(p, *args)

    def callback(p):
        eval_count['n'] += 1
        history_pass.append(eval_count['n'])
        history_obj.append(full_objective(p, train_x, train_y, d, K, C))
        history_err.append(word_error(p, test_x, test_y, d, K))
        print(
            f"  [LBFGS] step={eval_count['n']:4} "
            f"obj={history_obj[-1]:.4f} "
            f"test_err={history_err[-1]:.4f}"
        )


    params_opt, nfeval, rc = fmin_tnc(
        func=lambda p, *args: tracked_obj(p, *args),
        x0=params,
        args=(train_x, train_y, d, K, C),
        callback=callback,
        maxfun=maxfun
    )

    return params_opt, history_pass, history_obj, history_err

def plot_histories(sgd_hist, mom_hist, lbfgs_hist):
    _, sgd_pass, sgd_obj, sgd_err = sgd_hist
    _, mom_pass, mom_obj, mom_err = mom_hist
    _, lbfgs_pass, lbfgs_obj, lbfgs_err = lbfgs_hist

    plt.figure(figsize=(7,5))
    plt.plot(sgd_pass, sgd_obj, label="SGD")
    plt.plot(mom_pass, mom_obj, label="Momentum")
    plt.plot(lbfgs_pass, lbfgs_obj, label="LBFGS")
    plt.xlabel("Effective number of passes")
    plt.ylabel("Training objective")
    plt.title("Q4a: Training objective")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../result/q4a_objective.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7,5))
    plt.plot(sgd_pass, sgd_err, label="SGD")
    plt.plot(mom_pass, mom_err, label="Momentum")
    plt.plot(lbfgs_pass, lbfgs_err, label="LBFGS")
    plt.xlabel("Effective number of passes")
    plt.ylabel("Test word-wise error")
    plt.title("Q4a: Test word-wise error")
    plt.legend()
    plt.tight_layout()
    plt.savefig("../result/q4a_word_error.png", dpi=200)
    plt.close()

    '''
    # 1) SGD + Momentum: training objective vs effective passes
    plt.figure(figsize=(7, 5))
    plt.plot(sgd_pass, sgd_obj, label="SGD")
    plt.plot(mom_pass, mom_obj, label="Momentum")
    plt.xlabel("Effective number of passes")
    plt.ylabel("Training objective")
    plt.title("Q4a: Training objective (SGD vs Momentum)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("../result/q4a_objective_sgd_momentum.png", dpi=200)
    plt.close()

    # 2) LBFGS: training objective vs function evaluations
    plt.figure(figsize=(7, 5))
    plt.plot(lbfgs_eval, lbfgs_obj, label="LBFGS")
    plt.xlabel("Number of function evaluations")
    plt.ylabel("Training objective")
    plt.title("Q4a: Training objective (LBFGS)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("../result/q4a_objective_lbfgs.png", dpi=200)
    plt.close()

    # 3) SGD + Momentum: test word-wise error vs effective passes
    plt.figure(figsize=(7, 5))
    plt.plot(sgd_pass, sgd_err, label="SGD")
    plt.plot(mom_pass, mom_err, label="Momentum")
    plt.xlabel("Effective number of passes")
    plt.ylabel("Test word-wise error")
    plt.title("Q4a: Test word-wise error (SGD vs Momentum)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("../result/q4a_word_error_sgd_momentum.png", dpi=200)
    plt.close()

    # 4) LBFGS: test word-wise error vs function evaluations
    plt.figure(figsize=(7, 5))
    plt.plot(lbfgs_eval, lbfgs_err, label="LBFGS")
    plt.xlabel("Number of function evaluations")
    plt.ylabel("Test word-wise error")
    plt.title("Q4a: Test word-wise error (LBFGS)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("../result/q4a_word_error_lbfgs.png", dpi=200)
    plt.close()
    '''

# q4a
train_x, train_y = load_train("../data/train.txt")
test_x, test_y = load_train("../data/test.txt")

C = 1000
n = 2000

sgd_hist = run_sgd(train_x, train_y, test_x, test_y, 5, C=C, B=32, lr=1e-4, steps=n)
mom_hist = run_sgd_momentum(train_x, train_y, test_x, test_y, 5, C=C, B=32, lr=1e-4, momentum=0.9, steps=n)
lbfgs_hist = run_lbfgs(train_x, train_y, test_x, test_y, C=C, maxfun=n)
plot_histories(sgd_hist, mom_hist, lbfgs_hist)
