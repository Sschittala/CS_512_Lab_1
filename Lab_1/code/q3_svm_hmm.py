# This file uses the downloaded svm_hmm binaries to run SVM-HMM models 
# trained on *_struct.txt data with varying -c values
# and compares prediction values with true labels to evaluate letter-wise accuracy and plot for each C

import subprocess
import numpy as np
import matplotlib.pyplot as plt

# paths to binaries
svm_learn_path = "svm_hmm/svm_hmm_learn"
svm_classify_path = "svm_hmm/svm_hmm_classify"

# paths to input data
train_file = "../data/train_struct.txt"
test_file = "../data/test_struct.txt"

# paths to store model (to reference when predicting on test data)
model_file = "../result/svm_hmm-model_struct.txt"
predictions_file = "../result/svm_hmm-predictions_struct.txt"

def run_command(command_list):
    '''helper function (chatgpt) to help run commands and debug'''
    try:
        result = subprocess.run(
            command_list,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print(result.stdout)
        if result.stderr:
            print("Warnings / Info:\n", result.stderr)
    except subprocess.CalledProcessError as e:
        print("Error executing:", " ".join(command_list))
        print("Return code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)
        exit(1)

def read_labels(file_path):
    """read labels from svm-hmm formatted file (_struct.txt files)."""
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            # first token is the label
            label_str = line.split()[0]
            labels.append(int(label_str))
    return np.array(labels)

def letter_accuracy(y_true_seq, y_pred_seq):
    """
    y_true_seq, y_pred_seq: lists of arrays (each array = letters of a word)
    returns: letter-wise accuracy to use for evaluation
    """
    total_letters = 0
    correct_letters = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        total_letters += len(y_true)
        correct_letters += np.sum(y_true == y_pred)
    return correct_letters / total_letters

# extract labels to compare in test
y_test = read_labels(test_file)

# C-search values
C_values = [1, 10, 100, 1000]
letter_accuracies = []

print("starting C-search sweep for SVM-HMM...")
for C in C_values:
    # train model with current C value
    print(f"\ntraining with C={C} ...")
    run_command([svm_learn_path, "-c", str(C), train_file, model_file])
    
    # predict on test set
    print(f"predicting with C={C} ...")
    run_command([svm_classify_path, test_file, model_file, predictions_file])
    
    # compute letter-wise accuracy
    y_pred = read_labels(predictions_file)
    acc = letter_accuracy([y_test], [y_pred])
    letter_accuracies.append(acc)
    print(f"letter-wise accuracy for C={C}: {acc:.4f}")

# plot letter-wise accuracy vs C (help from chatgpt)
plt.figure(figsize=(6, 4))
plt.plot(C_values, letter_accuracies, marker='o')
plt.xscale('log')  # C is varied logarithmically
plt.xlabel("Regularization Parameter C")
plt.ylabel("Letter-wise Accuracy")
plt.title("SVM-Struct (SVM-HMM): Letter-wise Accuracy vs C")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()