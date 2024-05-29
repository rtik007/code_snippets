'''
This below code helps you extract and print human-readable rules from the Isolation Forest model trained using H2O.
Adjustments might be necessary based on the specifics of your dataset and model configuration.

'''

## Step 1: Initialize H2O and Train the Isolation Forest Model

import h2o
from h2o.estimators import H2OIsolationForestEstimator

# Initialize H2O
h2o.init()

# Generate a sample dataset and convert to H2OFrame
from sklearn.datasets import make_classification
import pandas as pd

X, _ = make_classification(n_samples=100, n_features=5, random_state=42)
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
h2o_df = h2o.H2OFrame(df)

# Train Isolation Forest
isolation_forest = H2OIsolationForestEstimator(ntrees=100, seed=42)
isolation_forest.train(training_frame=h2o_df)

## Step 2: Extract and Print Rules from Isolation Forest Trees

def traverse_tree(tree, feature_names):
    """
    Traverse an H2O tree and extract rules.
    """
    rules = []

    def recurse(node, depth, path):
        if node["is_leaf"]:
            rule = " and ".join(path)
            rules.append(f"If {rule} then anomaly")
        else:
            name = feature_names[node["split_feature"]]
            threshold = node["split_threshold"]
            left_path = path + [f"{name} <= {threshold}"]
            right_path = path + [f"{name} > {threshold}"]
            recurse(node["left_child"], depth + 1, left_path)
            recurse(node["right_child"], depth + 1, right_path)

    recurse(tree, 0, [])
    return rules

# Accessing the trees from the Isolation Forest model
trees = isolation_forest._model_json['output']['native_parameters']['trees']

# Extract rules from each tree
all_rules = []
for tree in trees:
    tree_structure = h2o.as_list(h2o.H2OTree(tree), use_pandas=True)
    feature_names = h2o_df.names
    for idx, row in tree_structure.iterrows():
        rules = traverse_tree(row, feature_names)
        all_rules.extend(rules)

# Print extracted rules
for rule in all_rules:
    print(rule)
