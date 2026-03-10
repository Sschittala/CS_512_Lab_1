# This file uses the provided liblibear package (SVM-MC models) to run models 
# trained on *_mc.txt data with varying -c values
# and compares prediction values with true labels to evaluate letter-wise accuracy and plot for each C.

from liblinear.liblinearutil import *
import numpy as np
import matplotlib.pyplot as plt

# file paths
train_file_mc = "../data/train_mc.txt"
test_file_mc = "../data/test_mc.txt"
original_test_file = "../data/test.txt"  # contains word IDs for word-wise

# letter-wise metric
def letter_accuracy(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sum(y_true == y_pred) / len(y_true)

# word-wise metric
def word_accuracy(y_true_words, y_pred_words):
    total_words = len(y_true_words)
    correct_words = sum(np.array_equal(t, p) for t, p in zip(y_true_words, y_pred_words))
    return correct_words / total_words

def split_words(labels, word_lengths):
    """Convert flat label list into list of words given word lengths"""
    words = []
    idx = 0
    for length in word_lengths:
        words.append(np.array(labels[idx:idx+length]))
        idx += length
    return words

def get_word_lengths_from_original(file_path):
    """Read original test.txt and compute number of letters per word"""
    word_lengths = []
    current_word = None
    count = 0
    with open(file_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if not tokens:
                continue
            word_id = int(tokens[3])  # 4th field is word_id
            if current_word is None:
                current_word = word_id
                count = 1
            elif word_id == current_word:
                count += 1
            else:
                word_lengths.append(count)
                current_word = word_id
                count = 1
        if count > 0:
            word_lengths.append(count)
    return word_lengths

# load in data
y_train, x_train = svm_read_problem(train_file_mc)
y_test, x_test = svm_read_problem(test_file_mc)
y_test_arr = np.array(y_test)

# compute word lengths and split true labels for word-wise evaluation
test_word_lengths = get_word_lengths_from_original(original_test_file)
y_test_words = split_words(y_test_arr, test_word_lengths)

print("training samples:", len(y_train))
print("test samples:", len(y_test))
print("number of test words:", len(y_test_words))

# c values
C_values = [1, 10, 100, 1000, 10000]
letter_accuracies = []
word_accuracies = []

for C in C_values:
    print(f"\ntraining with C={C} ...")
    model = train(y_train, x_train, f"-c {C}")

    print(f"predicting with C={C} ...")
    y_pred, _, _ = predict(y_test, x_test, model, options="-q")
    y_pred_arr = np.array(y_pred)
    y_pred_words = split_words(y_pred_arr, test_word_lengths)

    # compute accuracies
    acc_letter = letter_accuracy(y_test_arr, y_pred_arr)
    acc_word = word_accuracy(y_test_words, y_pred_words)

    letter_accuracies.append(acc_letter)
    word_accuracies.append(acc_word)

    print(f"Letter-wise accuracy: {acc_letter:.4f}")
    print(f"Word-wise accuracy:   {acc_word:.4f}")

# letter-wise
plt.figure(figsize=(6,4))
plt.plot(C_values, letter_accuracies, marker='o', color='blue')
plt.xscale('log')
plt.xlabel("Regularization Parameter C")
plt.ylabel("Letter-wise Accuracy")
plt.title("SVM-MC: Letter-wise Accuracy vs C")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()

# word-wise
plt.figure(figsize=(6,4))
plt.plot(C_values, word_accuracies, marker='s', color='blue')
plt.xscale('log')
plt.xlabel("Regularization Parameter C")
plt.ylabel("Word-wise Accuracy")
plt.title("SVM-MC: Word-wise Accuracy vs C")
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.show()