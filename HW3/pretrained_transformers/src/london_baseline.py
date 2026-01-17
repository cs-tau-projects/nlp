# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.
# Your solution here should only be a few lines.

import utils

# Count the number of examples in the dev set
eval_path = '../birth_dev.tsv'
with open(eval_path, encoding='utf-8') as f:
    num_examples = len(f.readlines())

# Create predictions list with "London" for every example
predictions = ["London"] * num_examples

# Evaluate using the existing function
total, correct = utils.evaluate_places(eval_path, predictions)
print(f"London baseline accuracy: {correct}/{total} = {correct/total*100:.2f}%")
