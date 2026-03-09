# This file uses the provided liblibear package (SVM-MC models) to run models 
# trained on *_mc.txt data with varying -c values
# and compares prediction values with true labels to evaluate letter-wise accuracy and plot for each C.

from liblinear.liblinearutil import *
import numpy as np
import matplotlib.pyplot as plt

# data file paths
train_file = "../data/train_mc.txt"
test_file = "../data/test_mc.txt"

def letter_accuracy(y_true_seq, y_pred_seq):
    """
    y_true_seq, y_pred_seq: lists of arrays (each array = letters of a word)
    returns: letter-wise accuracy
    """
    total_letters = 0
    correct_letters = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        total_letters += len(y_true)
        correct_letters += np.sum(y_true == y_pred)
    return correct_letters / total_letters

# use library to read in data
y_train, x_train = svm_read_problem(train_file)
y_test, x_test = svm_read_problem(test_file)
y_test_arr = np.array(y_test) # need to convert for using

print("training samples:", len(y_train))
print("test samples:", len(y_test))

# C-search values
C_values = [1, 10, 100, 1000]
letter_accuracies = []

print("\nstarting C-search sweep...")
for C in C_values:
    print(f"\ntraining with C={C} ...")
    model_c = train(y_train, x_train, f"-c {C}")

    print(f"\predicting with C={C} ...")
    pred_labels_c, _, _ = predict(y_test, x_test, model_c, options="-q")

    pred_labels_c_arr = np.array(pred_labels_c)
    acc_c = letter_accuracy([y_test_arr], [pred_labels_c_arr])
    letter_accuracies.append(acc_c)
    print(f"letter-wise accuracy for C={C}: {acc_c:.4f}")

# plot letter-wise accuracy vs C (help from chatgpt)
plt.figure(figsize=(6, 4))
plt.plot(C_values, letter_accuracies, marker='o')
plt.xscale('log')  # C is varied logarithmically
plt.xlabel("Regularization Parameter C")
plt.ylabel("Letter-wise Accuracy")
plt.title("SVM-MC: Letter-wise Accuracy vs C")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()