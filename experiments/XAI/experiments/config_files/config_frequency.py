# Let's create a script to create yaml files for execute the experiments
# We will use the same structure as the one used in the paper

import yaml
import os
from itertools import product
import tqdm

base_path = "../Frequency/experiments/"

if not os.path.exists(base_path):
    os.makedirs(base_path)

# path = "/home/patrick/Documents/Repositories/hiaac-m4-experiments/preliminary_analysis/datasets_preprocessing/data/har"

views = ["standartized_inter_balanced"]
standartized_views = ["standartized_inter_balanced"]
config_base = {
    "estimators": [
        {
            "algorithm": "RandomForest",
            "kwargs": {
                "n_estimators": 100,
                "random_state": 42,
            },
            "name": "random_forest-100",
            "num_runs": 1,
        },
        {
            "algorithm": "KNN",
            "kwargs": {
                "n_neighbors": 5,
            },
            "name": "knn-5",
            "num_runs": 1,
        },
        {
            "algorithm": "SVM",
            "kwargs": {
                "kernel": "rbf",
                "C": 1.0,
            },
            "name": "svm-rbf-C1.0",
            "num_runs": 1,
        },
        {
            "algorithm": "DecisionTree",
            "kwargs": {
                "criterion": "gini",
                "random_state": 42,
            },
            "name": "decision_tree-gini",
            "num_runs": 1,
        }
    ],
    "extra": {
        "in_use_features": [
            "accel-x",
            "accel-y",
            "accel-z",
            "gyro-x",
            "gyro-y",
            "gyro-z",
        ],
        "scale_on": "train",
        "reduce_on": "all",
        "save_data": True,
    },
    "transforms":
    [
        {
            "transform": "fft",
            "kwargs": {
                "centered": True,
                "absolute": True,
            },
            "name": "fft",
            "windowed": {
                "fit_on": None,
                "transform_on": "window",
            },
        },
    ],
    "reducer": None,
    "reducer_dataset": None,
    "scaler": None,
    "version": "1.0",
}

datasets = [
    "kuhar",
    "motionsense",
    "wisdm",
    "uci",
    "realworld_thigh",
    "realworld_waist",
]

cont = 0

experiments = list(
    product(
        datasets,    # 0 - dataset train
        views,       # 1 - view
        datasets,    # 2 - dataset test
    )
)


cont = 0
for args in tqdm.tqdm(experiments):
    config = config_base.copy()

    train = args[0]
    view = args[1]
    test = args[2]

    new_base_path = base_path
    os.mkdir(new_base_path) if not os.path.exists(new_base_path) else new_base_path

    config["train_dataset"] = [
        f"{train}.{view}[train]",
    ]
    config["validation_dataset"] = None
    config["test_dataset"] = [f"{test}.{view}[test]"]

    config["extra"]["scale_on"] = "train"

    cont += 1
    # Let's save the config file and the name will be the number of the config like 01, 02, ..., 10
    # config_path = os.path.join(new_base_path, f"{cont:04d}.yaml")
    config_path = os.path.join(new_base_path, f"{train}_{test}_freq.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
