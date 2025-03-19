# Let's create a script to create yaml files for execute the experiments
# We will use the same structure as the one used in the paper

import yaml
import os
from itertools import product
import tqdm

base_path = "../UMAP_axis/experiments/"

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

datasets_reducer = [
    ["motionsense", "uci", "wisdm"],
]

transforms = [
    # None,
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
    {
        "algorithm": "umap",
        "kwargs": {
            "n_components": 24,
            "random_state": 42,
        },
    },
]

reduce_on = ["axis"]
dimensions = [24]
cont = 0

experiments = list(
    product(
        transforms,  # 0 - transform
        datasets,    # 1 - dataset train
        views,       # 2 - view
        reduce_on,   # 3 - reduce_on
        reducer,     # 4 - reducer
        dimensions,  # 5 - dimensions
        datasets_reducer, # 6 - datasets_reducer
        datasets,    # 7 - dataset test
    )
)

invalid_combinations = [experiment 
                        for experiment in experiments 
                        if (experiment[2] not in standartized_views and experiment[4] != None) # Só pode ter reducer em standartized_{version}
                        or (experiment[3] != 'all' and experiment[4] == None) # Se não for all, tem que ter reducer

                        or (experiment[4] == None and experiment[5] != None) # Se não tiver reducer, não pode ter dimension
                        or (experiment[4] == None and experiment[6] != None) # Se não tiver reducer, não pode ter datasets_reducer
                        or (experiment[4] != None and experiment[5] == None) # Se tiver reducer, tem que ter dimension
                        or (experiment[4] != None and experiment[6] == None) # Se tiver reducer, tem que ter datasets_reducer
                    ]

experiments = [e for e in experiments if e not in invalid_combinations]

# cont_umap = 0
# cont_pca = 0
# cont = 0
# for e in experiments:
#     if e[4] != None:
#         cont_umap += 1 if e[4]['name'] == 'umap' else 0
#         cont_pca += 1 if e[4]['name'] == 'pca' else 0
#         cont += 1
        
# print(f"Total of experiments with reducer: UMAP  - {cont_umap}, PCA - {cont_pca}, Total - {cont}")

cont = 0
for args in tqdm.tqdm(experiments):
    config = config_base.copy()

    transform = args[0]
    train = args[1]
    view = args[2]
    reduce_on = args[3]
    reducer = args[4]
    dimension = args[5]
    datasets_reducer = args[6]
    test = args[7]

    new_base_path = base_path
    os.mkdir(new_base_path) if not os.path.exists(new_base_path) else new_base_path

    if view == "initial_dataset" and transform is not None:
        transform[0]['kwargs']['absolute'] = False
    else:
        if transform is not None:
            transform[0]['kwargs']['absolute'] = True

    config["transforms"] = transform

    config["train_dataset"] = [
        f"{train}.{view}[train]",
    ]
    config["validation_dataset"] = None
    config["test_dataset"] = [f"{test}.{view}[test]"]

    config["extra"]["scale_on"] = "train"

    if reducer is not None:
        config["reducer_dataset"] = [dataset+".standartized_inter_balanced[train]" for dataset in datasets_reducer if dataset is not None]
        
        config["extra"]["reduce_on"] = reduce_on
        dim = int(dimension if reduce_on == 'all' else dimension // 2 if reduce_on == 'sensor' else dimension // 6)
        # dim = 24 if reduce_on == 'all' else 12 if reduce_on == 'sensor' else 4
        # print(reducer)
        config['reducer'] = reducer
        config['reducer']['kwargs']['n_components'] = dim
        config['reducer']['name'] = f"{reducer['algorithm']}-{dim}"

    cont += 1
    # Let's save the config file and the name will be the number of the config like 01, 02, ..., 10
    # config_path = os.path.join(new_base_path, f"{cont:04d}.yaml")
    config_path = os.path.join(new_base_path, f"{train}_{test}_{reduce_on}.yaml")
    with open(config_path, "w") as f:
        yaml.dump(config, f)