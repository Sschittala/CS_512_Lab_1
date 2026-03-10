# This file uses the svm_hmm binaries to run models 
# trained on *_struct.txt data with varying -c values
# and compares prediction values with true labels to evaluate letter-wise accuracy and plot for each C.

import subprocess
import numpy as np
import matplotlib.pyplot as plt

# paths to binaries
svm_learn_path = "svm_hmm/svm_hmm_learn"
svm_classify_path = "svm_hmm/svm_hmm_classify"

# paths to input data
train_file = "../data/train_struct.txt"
test_file = "../data/test_struct.txt"

# paths to store model and predictions
model_file = "../result/svm_hmm-model_struct.txt"
predictions_file = "../result/svm_hmm-predictions_struct.txt"

# calculate letter accuracy
def letter_accuracy(y_true_seq, y_pred_seq):
    total_letters = sum(len(w) for w in y_true_seq)
    correct_letters = sum(np.sum(w_true == w_pred) for w_true, w_pred in zip(y_true_seq, y_pred_seq))
    return correct_letters / total_letters

# calculate word accuracy
def word_accuracy(y_true_seq, y_pred_seq):
    total_words = len(y_true_seq)
    correct_words = sum(np.array_equal(w_true, w_pred) for w_true, w_pred in zip(y_true_seq, y_pred_seq))
    return correct_words / total_words

# helper function to run binaries and debug
def run_command(command_list):
    """Run a shell command and print output/errors."""
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

# extract labels from svm-hmm formatted _struct.txt files
def read_labels(file_path):
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                label = int(line.split()[0])
                labels.append(label)
    return labels

# group flat label list into words using the test set word lengths
def read_labels_by_word(flat_labels, words_lengths):
    words = []
    idx = 0
    for length in words_lengths:
        words.append(np.array(flat_labels[idx: idx + length]))
        idx += length
    return words

# extract word lengths from _struct
def get_word_lengths(file_path):
    word_lengths = []
    current_qid = None
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            qid = int(tokens[1].split(":")[1])
            if current_qid is None:
                current_qid = qid
                count = 1
            elif qid == current_qid:
                count += 1
            else:
                word_lengths.append(count)
                current_qid = qid
                count = 1
        # append last word
        if count > 0:
            word_lengths.append(count)
    return word_lengths

# get word lengths from test set
test_word_lengths = get_word_lengths(test_file)
print("Number of words in test set:", len(test_word_lengths))
print("Number of letters in test set:", sum(test_word_lengths))

# extract true labels from test set
y_test_flat = read_labels(test_file)
y_test_words = read_labels_by_word(y_test_flat, test_word_lengths)

# C-search values
C_values = [1, 10, 100, 1000, 10000]
letter_accuracies = []
word_accuracies = []

print("starting C-search sweep for SVM-HMM...")
for C in C_values:
    print(f"\ntraining SVM-HMM with C={C} ...")
    run_command([svm_learn_path, "-c", str(C), train_file, model_file])
    
    print(f"predicting with C={C} ...")
    run_command([svm_classify_path, test_file, model_file, predictions_file])
    
    # read predicted labels
    y_pred_flat = read_labels(predictions_file)
    y_pred_words = read_labels_by_word(y_pred_flat, test_word_lengths)
    
    # compute accuracies
    letter_acc = letter_accuracy(y_test_words, y_pred_words)
    word_acc = word_accuracy(y_test_words, y_pred_words)
    
    letter_accuracies.append(letter_acc)
    word_accuracies.append(word_acc)
    
    print(f"letter-wise accuracy for C={C}: {letter_acc:.4f}")
    print(f"word-wise accuracy for C={C}: {word_acc:.4f}")


# plotting with assistance from ChatGPT

# q3_svm_hmm_letter-wise.png
plt.figure(figsize=(6,4))
plt.plot(C_values, letter_accuracies, marker='o')
plt.xscale('log')
plt.xlabel("Regularization Parameter C")
plt.ylabel("Letter-wise Accuracy")
plt.title("SVM-HMM Letter-wise Accuracy vs C")
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.show()

# q3_svm_hmm_word-wise.png
plt.figure(figsize=(6,4))
plt.plot(C_values, word_accuracies, marker='o')
plt.xscale('log')
plt.xlabel("Regularization Parameter C")
plt.ylabel("Word-wise Accuracy")
plt.title("SVM-HMM Word-wise Accuracy vs C")
plt.grid(True, which="both", linestyle="--")
plt.tight_layout()
plt.show()