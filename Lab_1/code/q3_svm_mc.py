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
    total_letters = 0
    correct_letters = 0

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        total_letters += len(y_true)
        correct_letters += np.sum(y_true == y_pred)

    return correct_letters / total_letters

def word_accuracy(y_true_seq, y_pred_seq):
    total_words = len(y_true_seq)
    correct_words = 0

    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        if np.array_equal(y_true, y_pred):
            correct_words += 1

    return correct_words / total_words

def split_words(labels, word_lengths):
    """Convert flat label list into a list of words given word lengths."""
    words = []
    idx = 0
    for length in word_lengths:
        words.append(np.array(labels[idx:idx+length]))
        idx += length
    return words

def get_word_lengths(file_path):
    """Assume sequential letters for words in MC file; compute lengths of each word from qid fields."""
    word_lengths = []
    current_qid = None
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            # MC format uses qid:X to indicate word
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
        if count > 0:
            word_lengths.append(count)
    return word_lengths

# use library to read in data
y_train, x_train = svm_read_problem(train_file)
y_test, x_test = svm_read_problem(test_file)
y_test_arr = np.array(y_test) # need to convert for using

print("training samples:", len(y_train))
print("test samples:", len(y_test))

# compute test word lengths for word-wise accuracy
test_word_lengths = get_word_lengths(test_file)
y_test_words = split_words(y_test_arr, test_word_lengths)

# C-search values
C_values = [1, 10, 100, 1000, 10000]
letter_accuracies = []
word_accuracies = []

print("\nstarting C-search sweep...")
for C in C_values:
    print(f"\ntraining with C={C} ...")
    model_c = train(y_train, x_train, f"-c {C}")

    print(f"\npredicting with C={C} ...")
    pred_labels_c, _, _ = predict(y_test, x_test, model_c, options="-q")
    pred_labels_c_arr = np.array(pred_labels_c)
    pred_labels_words = split_words(pred_labels_c_arr, test_word_lengths)

    # compute accuracies
    acc_c_letter = letter_accuracy([y_test_arr], [pred_labels_c_arr])
    acc_c_word = word_accuracy(y_test_words, pred_labels_words)
    letter_accuracies.append(acc_c_letter)
    word_accuracies.append(acc_c_word)

    print(f"letter-wise accuracy for C={C}: {acc_c_letter:.4f}")
    print(f"word-wise accuracy for C={C}: {acc_c_word:.4f}")

# plot letter-wise accuracy vs C
plt.figure(figsize=(6, 4))
plt.plot(C_values, letter_accuracies, marker='o')
plt.xscale('log')  # C is varied logarithmically
plt.xlabel("Regularization Parameter C")
plt.ylabel("Letter-wise Accuracy")
plt.title("SVM-MC: Letter-wise Accuracy vs C")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# plot word-wise accuracy vs C
plt.figure(figsize=(6, 4))
plt.plot(C_values, word_accuracies, marker='o')
plt.xscale('log')  # C is varied logarithmically
plt.xlabel("Regularization Parameter C")
plt.ylabel("Word-wise Accuracy")
plt.title("SVM-MC: Word-wise Accuracy vs C")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()