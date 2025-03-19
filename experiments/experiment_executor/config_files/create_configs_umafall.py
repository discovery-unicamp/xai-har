# Let's create a script to create yaml files for execute the experiments
# We will use the same structure as the one used in the paper

import yaml
import os
from itertools import product
import tqdm

base_path = "../fall_experiments/experiments/"

if not os.path.exists(base_path):
    os.makedirs(base_path)

path = "/home/patrick/Documents/Repositories/hiaac-m4-experiments/preliminary_analysis/datasets_preprocessing/data/fall/standartized_balanced/UMAFall"
# users are the file names in path
folds = sorted(os.listdir(path))

# views = ["raw_balanced", "standartized_balanced"]
views = ["raw_balanced", "standartized_balanced"]

config_base = {
    "estimators": [
        # {
        #     "algorithm": "KNN",
        #     "kwargs": {
        #         "n_neighbors": 5,
        #     },
        #     "name": "knn-5",
        #     "num_runs": 1,
        # },
        {
            "algorithm": "OC-SVM",
            "kwargs": {
                "kernel": "poly",
                "degree": 3,
                "nu": 0.001,
            },
            "name": "oc-svm",
            "num_runs": 1,
        },
        {
            "algorithm": "IsolationForest",
            "kwargs": {
                "contamination": 0.1,
                "random_state": 42,
            },
            "name": "isolation-forest",
            "num_runs": 1,
        }
    ],
    "extra": {
        "in_use_features": [
            "accel-x",
            "accel-y",
            "accel-z",
        ],
        "scale_on": "train",
        "reduce_on": "all",
    },
    "reducer": None,
    "reducer_dataset": None,
    "scaler": None,
    "version": "1.0",
}

datasets = [
    "umafall",
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
            "n_components": 12,
            "random_state": 42,
        },
        "name": "umap-12",
    },
    {
        "algorithm": "pca",
        "kwargs": {
            "n_components": 12,
            "random_state": 42,
        },
        "name": "pca-12",
    }
]

reduce_on = ["all", "axis"]
# reduce_on = ["all"]

cont = 0

experiments = list(product(
    transforms,  # 0
    datasets,    # 1
    folds,       # 2
    views,       # 3
    reduce_on,   # 4
    reducer      # 5
))

# invalid_combinations = [experiment 
#                         for experiment in experiments 
#                         if (experiment[3].find("standartized") != -1 and experiment[5] != None) # Só pode ter reducer em standartized_{version}
#                         or (experiment[4] != 'all' and experiment[5] == None) # Se não for all, tem que ter reducer
#                         ]
                        
# experiments = [e for e in experiments if e not in invalid_combinations]
for args in tqdm.tqdm(experiments):
    config = config_base.copy()

    transform = args[0]
    dataset = args[1]
    fold = args[2]
    view = args[3]
    reduce_on = args[4]
    reducer = args[5]


    config["transforms"] = transform

    config["train_dataset"] = [
        f"{dataset}.{view}[train]",
    ]
    config["validation_dataset"] = None
    config["test_dataset"] = [f"{dataset}.{view}[test]"]

    config["dataset_structure"] = {
        "type": "k-fold",
        "fold": f"{fold}",
        "load_train": True,
        "load_validation": False,
        "load_test": True,
    }

    config["extra"]["in_use_features"] = [
        "accel-x",
        "accel-y",
        "accel-z",
    ]

    config["extra"]["scale_on"] = "train"

    if reducer is not None and view == "standartized_balanced":
        config["reducer"] = reducer
        config["reducer_dataset"] = [
            f"kuhar.standartized_inter_balanced[train]",
            f"motionsense.standartized_inter_balanced[train]",
            f"wisdm.standartized_inter_balanced[train]",
        ]
        config["extra"]["reduce_on"] = reduce_on

        if reduce_on == "axis":
            config["reducer"]["kwargs"]["n_components"] = 4
    

    cont += 1
    # Let's save the config file and the name will be the number of the config like 01, 02, ..., 10
    config_path = os.path.join(base_path, f"{cont:04d}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)
