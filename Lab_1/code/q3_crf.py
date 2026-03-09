# This file uses the CRF model developed in q2_gradient.py
# trained on train.txt data with varying -c values
# and compares prediction values with true labels to evaluate letter-wise accuracy and plot for each C

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_tnc

# reuse functions from our CRF model
from q1_decode import decode_dp
from q2_gradient import load_train, objective_and_grad

def letter_accuracy(y_true_seq, y_pred_seq):
    total_letters = 0
    correct_letters = 0

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        total_letters += len(y_true)
        correct_letters += np.sum(y_true == y_pred)

    return correct_letters / total_letters

# load in data
train_x, train_y = load_train("../data/train.txt")
test_x, test_y = load_train("../data/test.txt")

print("training words:", len(train_x))
print("test words:", len(test_x))

# C-search values
C_values = [1, 10, 100, 1000]
letter_accuracies = []

print("starting C-search sweep for CRF...")
for C in C_values:

    print(f"\ntraining CRF with C={C}")

    W_init = np.zeros((26, 128))
    T_init = np.zeros((26, 26))

    params_init = np.concatenate([
        W_init.flatten(),
        T_init.flatten('F')
    ])

    solution = fmin_tnc(
        func=lambda p, *args: objective_and_grad(p, *args),
        x0=params_init,
        args=(train_x, train_y, 128, 26, C),
        messages=0
    )

    params_opt = solution[0]

    W_opt = params_opt[:26*128].reshape(26, 128)
    T_opt = params_opt[26*128:].reshape(26, 26, order='F')

    print(f"\ntraining CRF with C={C}")
    # now predict and decode
    predictions = []
    for X in test_x:
        pred = decode_dp(X, W_opt, T_opt)

        # convert from 1..26 to 0..25
        predictions.append(pred - 1)

    # calculate accuracy
    acc = letter_accuracy(test_y, predictions)
    letter_accuracies.append(acc)

    print(f"letter-wise accuracy for C={C}: {acc:.4f}")

plt.figure(figsize=(6,4))
plt.plot(C_values, letter_accuracies, marker='o')
plt.xscale('log')

plt.xlabel("Regularization Parameter C")
plt.ylabel("Letter-wise Accuracy")
plt.title("CRF Letter-wise Accuracy vs C")

plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.show()