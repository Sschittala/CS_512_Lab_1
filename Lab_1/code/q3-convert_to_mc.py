
# This file converts the data files into a format that is readable by SVM-MC model (LibSVM module)

import os

# input files in SVM-hmm format
train_struct_file = "../data/train_struct.txt"
test_struct_file = "../data/test_struct.txt"

# output files in SVM-mc
train_mc_file = "../data/train_mc.txt"
test_mc_file = "../data/test_mc.txt"

# conversion function
def convert_to_libsvm(input_file, output_file):
    with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if not parts:
                continue
            label = parts[0]
            features = []
            for item in parts[1:]:
                if item.startswith("qid:"):
                    continue  # ignore sequence ID
                features.append(item)
            f_out.write(label + " " + " ".join(features) + "\n")

# call function to convert both files
convert_to_libsvm(train_struct_file, train_mc_file)
convert_to_libsvm(test_struct_file, test_mc_file)

print("conversion complete!")
print(f"train file written to: {train_mc_file}")
print(f"test file written to: {test_mc_file}")