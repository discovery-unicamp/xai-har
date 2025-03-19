# Let's create a script to create yaml files for execute the experiments
# We will use the same structure as the one used in the paper

import yaml
import os
from itertools import product
import tqdm

base_path = "../only_transitions_experiments_unbalanced"
if not os.path.exists(base_path):
    os.makedirs(base_path)
base_path = base_path + "/experiments"
if not os.path.exists(base_path):
    os.makedirs(base_path)

path = "/home/patrick/Documents/Repositories/hiaac-m4-experiments/preliminary_analysis/datasets_preprocessing/data/transitions/initial_dataset_unbalanced/HAPT_only_transitions"
# users are the file names in path
users = sorted(os.listdir(path))

views = ["initial_dataset_unbalanced", "raw_unbalanced", "standartized_unbalanced"]
# views = ["raw_balanced", "standartized_balanced"]

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
                "n_neighbors": 7,
            },
            "name": "knn-7",
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
        "label_columns": "standard activity code",
        "reduce_on": "all",
    },
    "reducer": None,
    "reducer_dataset": None,
    "scaler": None,
    "version": "1.0",
}

datasets = [
    "hapt_only_transitions",
]

transforms = [
    None,
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
]

reducer = [
    None,
    {
        "algorithm": "umap",
        "kwargs": {
            "n_components": 24,
            "random_state": 42,
        },
        "name": "umap",
    },
    {
        "algorithm": "pca",
        "kwargs": {
            "n_components": 24,
            "random_state": 42,
        },
        "name": "pca",
    }
]

reduce_on = ["all", "sensor", "axis"]

cont = 0

experiments = list(product(
    transforms,  # 0
    datasets,    # 1
    users,       # 2
    views,       # 3
    reduce_on,   # 4
    reducer,     # 5
))

invalid_combinations = [experiment 
                        for experiment in experiments 
                        if (experiment[3].find("standartized") != -1 and experiment[5] != None) # Só pode ter reducer em standartized_{version}
                        or (experiment[4] != 'all' and experiment[5] == None) # Se não for all, tem que ter reducer
                        ]

experiments = [e for e in experiments if e not in invalid_combinations]

for args in tqdm.tqdm(experiments):
    config = config_base.copy()

    transform = args[0]
    dataset = args[1]
    user = args[2]
    view = args[3]
    reduce_on = args[4]
    reducer = args[5]

    if transform is not None:
        transform[0]['kwargs']['absolute'] = True

    config["transforms"] = transform

    config["train_dataset"] = [
        f"{dataset}.{view}[train]",
    ]
    config["validation_dataset"] = None
    config["test_dataset"] = [f"{dataset}.{view}[test]"]

    config["dataset_structure"] = {
        "type": "user",
        "user_id": user,
        "fold": None,
        "load_train": True,
        "load_validation": False,
        "load_test": True,
    }

    config["extra"]["in_use_features"] = [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ]

    config["extra"]["scale_on"] = "train"
    config["reducer"] = reducer

    if reducer is not None:
        config["reducer_dataset"] = [
            f"kuhar.standartized_inter_balanced[train]",
            f"motionsense.standartized_inter_balanced[train]",
            f"wisdm.standartized_inter_balanced[train]",
        ]
        config["extra"]["reduce_on"] = reduce_on
        config['reducer']['kwargs']['n_components'] = 24 if reduce_on == 'all' else 12 if reduce_on == 'sensor' else 4
        config['reducer']['name'] = f"{reducer['name']}-{config['reducer']['kwargs']['n_components']}"

    cont += 1
    # Let's save the config file and the name will be the number of the config like 01, 02, ..., 10
    config_path = os.path.join(base_path, f"{cont:04d}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
