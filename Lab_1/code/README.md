# Code Folder Overview

This folder contains all the scripts, modules, and SVM-HMM implementation files used in Lab 1.

### Python Scripts
- **q1_decode.py** — Implements the decoding algorithm in brute force and max-sum. Used in other files to test results for CRF model. (1c)

- **q2_gradient.py** — This file implements gradient computation and training utilities for a letter-level Conditional Random Field (CRF), including forward-backward, gradient calculation, data loading, and saving solutions. Used throughout the Lab wherever the CRF model is needed. (2a, 2b)

- **q2_test.py** — Test script that trains a letter-level CRF on the provided dataset, inspects learned transition weights, and runs diagnostic checks on the last few words to verify the model’s computations.

- **q3_convert_to_mc.py** — Converts the structured training and test datasets into the LibSVM-compatible format required for running multi-class SVM experiments.

- **q3_crf.py** — Runs experiments of CRF model on the dataset for various c values, decodes predictions on test data using the learned weights, computes both letter- and word-level accuracies, and plots how accuracy changes with the regularization parameter C. (3a, 3b)

- **q3_svm_hmm.py** — Runs experiments of SVM-HMM model on the dataset for various c values, evaluates predictions on test data, computes both letter- and word-level accuracies, and plots how accuracy changes with the regularization parameter C. (3a, 3b)

- **q3_svm_mc.py** — Runs experiments of SVM-MC model on the dataset for various c values, evaluates predictions on test data, computes both letter- and word-level accuracies, and plots how accuracy changes with the regularization parameter C. (3a, 3b)

- **q4_sgd.py** — Implements stochastic gradient descent (SGD) and variants—including plain SGD, SGD with momentum, and L-BFGS—to train the CRF model, track training objectives and test word-wise errors over iterations, and visualize their progress with plots. (4a, 4b)

- **q4b_MCMC.py** — Implements MCMC-based training for the CRF model, using Gibbs sampling to approximate gradients, and provides functions to run SGD, SGD with momentum, and L-BFGS while tracking training objectives and test word-wise errors. (4b)

- **q4c_rao_blackwell.py** — Runs experiments comparing standard Gibbs sampling and Rao-Blackwellized Gibbs sampling for the CRF model, estimating node and edge marginals, computing KL divergence against the exact marginals, and plotting how the divergence decreases as the number of samples increases. (4c)

- **q5_driver.py** — Runs the transformation experiments: it applies increasing numbers of predefined transformations to the training letters, retrains both a CRF and an SVM-MC model on the transformed data, evaluates letter- and word-wise accuracies on the unchanged test set, and plots how these accuracies vary with the number of transformations applied. (5a, 5b)

- **q5_transform_util.py** — This file provides utility functions for Q5: it loads letter-level training data, applies pixel-level rotations or translations to letters based on a transformation dictionary, updates the dataset accordingly, and saves the transformed data back to a file. (5a, 5b)

- **ref_optimize.py** — Reference optimization routines.

### `svm_hmm/`
Contains the full SVM-HMM implementation:

- Can mostly ignore, only needed to call binary executables.

## Usage
1. Run the question-specific Python scripts to perform experiments.
2. Uncomment/Comment sections in the bottom of documents to run individual files as needed.
3. Use the SVM-HMM executables inside `svm_hmm/` for structured prediction tasks.
4. Helper modules like `q5_transform_util.py` provide common functions for data processing.