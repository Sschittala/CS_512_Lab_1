
# main driver file that executes q5. applies specified number of transformations to the train data,
# and records how both the CRF model and SVM-MC model scores change

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from liblinear.liblinearutil import train, predict, svm_read_problem
from scipy.optimize import fmin_tnc

# utils from all of previous questions
from q5_transform_util import load_train_data, transform_train_letters
from q3_convert_to_mc import convert_to_libsvm
from q2_gradient import load_train as load_crf_train, objective_and_grad
from q1_decode import decode_dp

# file paths
train_file = '../data/train.txt'
test_file = '../data/test.txt'
temp_train_file = '../data/train_transformed.txt'

# varying amount of transformations
trans_values = [0, 500, 1000, 1500, 2000]

# hyperparameters defined from q3
crf_best_C = 1000
svm_mc_best_C = 1

# load in the transformations
letters_orig = load_train_data(train_file)
with open('../data/transform.txt', 'r') as f:
    all_transforms = f.readlines()

# load in the test data, will remain unchanged
# records extra information for SVM-MC word metrics
y_true_flat = []
test_word_lengths = []
with open(test_file, 'r') as f:
    current_word = None
    count = 0
    for line in f:
        tokens = line.strip().split()
        if not tokens:
            continue
        letter = tokens[1]
        word_id = int(tokens[3])
        y_true_flat.append(ord(letter) - ord('a') + 1)  # 1..26 for SVM-MC
        if current_word is None:
            current_word = word_id
            count = 1
        elif word_id == current_word:
            count += 1
        else:
            test_word_lengths.append(count)
            current_word = word_id
            count = 1
    if count > 0:
        test_word_lengths.append(count)
y_true_flat = np.array(y_true_flat)

# load CRF formatted test data, will remain unchanged
test_x_crf, test_y_crf = load_crf_train(test_file)
y_true_crf_flat = np.array([label for word in test_y_crf for label in word])

# helper function to convert to _struct format (ChatGPT)
def save_transformed_struct(letters, output_file):
    """Save letters in LibSVM _struct format for SVM-MC."""
    with open(output_file, 'w') as f:
        for letter in letters:
            label = ord(letter['letter']) - ord('a') + 1
            features = [f"{i+1}:1" for i, val in enumerate(letter['pixels']) if val != 0]
            f.write(f"{label} " + " ".join(features) + "\n")

# helper function used when calculating word-wise for SVM-MC (ChatGPT)
def split_words(labels, word_lengths):
    """Split flat labels into words using word lengths."""
    words = []
    idx = 0
    for wl in word_lengths:
        words.append(np.array(labels[idx:idx+wl]))
        idx += wl
    return words


# main execution loop, record accuracies for each tier of transformation
crf_letter_acc = []
crf_word_acc = []
svm_letter_acc = []
svm_word_acc = []
for x in trans_values:
    print(f"\napplying first {x} transformations...")

    # apply first x transformations
    transform_dict = {}
    for line in all_transforms[:x]:
        parts = line.strip().split()
        if parts[0] == 'r':
            transform_dict[int(parts[1])] = ('r', float(parts[2]))
        elif parts[0] == 't':
            transform_dict[int(parts[1])] = ('t', int(parts[2]), int(parts[3]))

    letters_transformed = transform_train_letters(deepcopy(letters_orig), transform_dict)

    # save flattened transformed letters for CRF
    np.savetxt(temp_train_file, 
               [[l['id'], l['letter'], l['next_id'], l['word_id'], l['position'], *l['pixels']] 
                for l in letters_transformed], fmt='%s')

    # train CRF model using our q2 code (longer runtime)
    print("training CRF...")
    train_x_crf, train_y_crf = load_crf_train(temp_train_file)
    W_init = np.zeros((26, 128))
    T_init = np.zeros((26, 26))
    params_init = np.concatenate([W_init.ravel(), T_init.ravel()])

    # optimize
    sol = fmin_tnc(func=lambda p, *args: objective_and_grad(p, *args),
                   x0=params_init,
                   args=(train_x_crf, train_y_crf, 128, 26, crf_best_C),
                   maxfun=250,
                   ftol=1e-3,
                   messages=0)
    params_opt = sol[0]
    W_opt = params_opt[:26*128].reshape(26, 128)
    T_opt = params_opt[26*128:].reshape(26, 26)

    # predict CRF using our q1 decode code
    preds_crf = [decode_dp(X, W_opt, T_opt)-1 for X in test_x_crf]  # convert 1..26 -> 0..25
    crf_letter_acc.append(np.sum(y_true_crf_flat == np.array([l for w in preds_crf for l in w])) / len(y_true_crf_flat))
    crf_word_acc.append(np.mean([np.array_equal(w_true, w_pred) for w_true, w_pred in zip(test_y_crf, preds_crf)]))

    # need to convert transformed train data to _struct
    print("saving transformed letters in _struct format...")
    struct_file = '../data/train_transformed_struct.txt'
    save_transformed_struct(letters_transformed, struct_file)

    # then convert to _mc to be used
    mc_file = '../data/train_transformed_mc.txt'
    convert_to_libsvm(struct_file, mc_file)

    # train SVM-MC mode
    y_train_mc, x_train_mc = svm_read_problem(mc_file)
    y_test_mc, x_test_mc = svm_read_problem('../data/test_mc.txt')  # existing test_mc

    print("training SVM-MC...")
    model = train(y_train_mc, x_train_mc, f"-c {svm_mc_best_C}")

    # predict
    y_pred_mc, _, _ = predict(y_test_mc, x_test_mc, model, options='-q')
    y_pred_mc_arr = np.array(y_pred_mc)

    # letter-wise accuracy for SVM-MC
    acc_letter = np.sum(y_true_flat == y_pred_mc_arr) / len(y_true_flat)
    svm_letter_acc.append(acc_letter)

    # word-wise accuracy SVM-MC
    y_pred_words = split_words(y_pred_mc_arr, test_word_lengths)
    y_true_words = split_words(y_true_flat, test_word_lengths)
    acc_word = sum(np.array_equal(t, p) for t, p in zip(y_true_words, y_pred_words)) / len(test_word_lengths)
    svm_word_acc.append(acc_word)

    print(f"CRF letter-wise: {crf_letter_acc[-1]:.4f}, word-wise: {crf_word_acc[-1]:.4f}")
    print(f"SVM-MC letter-wise: {acc_letter:.4f}, word-wise: {acc_word:.4f}")

# Letter-wise Accuracy plot
plt.figure(figsize=(6,4))
plt.plot(trans_values, crf_letter_acc, marker='o', label='CRF', color='blue')
plt.plot(trans_values, svm_letter_acc, marker='s', label='SVM-MC', color='red')
plt.xlabel("Number of Transformations Applied")
plt.ylabel("Letter-wise Accuracy")
plt.title("Letter-wise Accuracy vs Transformations")
plt.legend()
plt.grid(True, ls='--')
plt.tight_layout()
plt.show()

# Word-wise Accuracy plot
plt.figure(figsize=(6,4))
plt.plot(trans_values, crf_word_acc, marker='o', label='CRF', color='blue')
plt.plot(trans_values, svm_word_acc, marker='s', label='SVM-MC', color='red')
plt.xlabel("Number of Transformations Applied")
plt.ylabel("Word-wise Accuracy")
plt.title("Word-wise Accuracy vs Transformations")
plt.legend()
plt.grid(True, ls='--')
plt.tight_layout()
plt.show()