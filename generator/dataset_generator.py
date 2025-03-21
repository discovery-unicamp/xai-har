from pathlib import Path
import os, shutil
import pandas as pd
from typing import Tuple, List, Dict, Any

from dataset_processor import (
    AddGravityColumn,
    AddSMV,
    ButterworthFilter,
    MedianFilter,
    CalcTimeDiffMean,
    Convert_G_to_Ms2,
    ResamplerPoly,
    Windowize,
    Peak_Windowize,
    AddStandardActivityCode,
    SplitGuaranteeingAllClassesPerSplit,
    BalanceToMinimumClass,
    BalanceToMinimumClassAndUser,
    LeaveOneSubjectOut,
    FilterByCommonRows,
    RenameColumns,
    Pipeline,
    GenerateFold,
)

# Set the seed for reproducibility
import numpy as np
import random

from utils import (
    read_kuhar,
    read_motionsense,
    read_wisdm,
    read_uci,
    read_realworld,
    read_recodGaitV1,
    read_recodGaitV2,
    read_GaitOpticalInertial,
    read_umafall,
    sanity_function,
    real_world_organize,
)

"""This module  used to generate the datasets. The datasets are generated in the following steps:    
    1. Read the raw dataset
    2. Preprocess the raw dataset
    3. Preprocess the standartized dataset
    4. Remove activities that are equal to -1
    5. Balance the dataset per activity
    6. Balance the dataset per user and activity
    7. Save the datasets
    8. Generate the views of the datasets
"""

random.seed(42)
np.random.seed(42)

# Variables used to map the activities from the RealWorld dataset to the standard activities
maping: List[int] = [4, 3, -1, -1, 5, 0, 1, 2]
tasks: List[str] = [
    "climbingdown",
    "climbingup",
    "jumping",
    "lying",
    "running",
    "sitting",
    "standing",
    "walking",
]
standard_activity_code_realworld_map: Dict[str, int] = {
    activity: maping[tasks.index(activity)] for activity in tasks
}

har_datasets: List[str] = [
    "KuHar",
    "MotionSense",
    "UCI",
    "WISDM",
    "RealWorld",
]

authentications_datasets: List[str] = [
    "RecodGait_v1",
    "RecodGait_v2",
    "GaitOpticalInertial",
]

fall_datasets: List[str] = ["UMAFall"]
transitions_datasets: List[str] = ["HAPT", 'HAPT_different_transitions', 'HAPT_only_transitions']

column_group: Dict[str, str] = {
    "KuHar": ["user", "activity code", "csv"],
    "MotionSense": ["user", "activity code", "csv"],
    "WISDM": ["user", "activity code", "window"],
    "UCI": ["user", "activity code", "serial"],
    "RealWorld": ["user", "activity code", "position"],
    "RecodGait_v1": ["user", "index", "session"],
    "RecodGait_v2": ["user", "index", "session"],
    "GaitOpticalInertial": ["user", "session"],
    "UMAFall": ["file_name"],
    "HAPT": ["user", "activity code", "serial"],
    "HAPT_different_transitions": ["user", "activity code", "serial"],
    "HAPT_only_transitions": ["user", "activity code", "serial"],
}

'''
The standard activity code has the following mapping:
    0: Sitting
    1: Standing
    2: Walking
    3: Stair Up
    4: Stair Down
    5: Running
    6: Stars up and down
    7: 
    8: Laying
    9: Stand to sit
    10: Sit to stand
    11: Sit to lie
    12: Lie to sit
    13: Stand to lie
    14: Lie to stand
    15: ADL
    16: Fall
'''

standard_activity_code_map: Dict[str, Dict[Any, int]] = {
    "KuHar": {
        0: 1,   # Standing
        1: 0,   # Sitting
        2: -1,  # Talk-sit
        3: -1,  # Talk-stand
        4: -1,  # Stand-sit
        5: -1,  # Lay
        6: -1,  # Lay-stand
        7: -1,  # Pick
        8: -1,  # Jump
        9: -1,  # Push-up
        10: -1, # Sit-up
        11: 2,  # Walk
        12: -1, # Walk-backwards
        13: -1, # Walk-circle
        14: 5,  # Run
        15: 3,  # Stairs up
        16: 4,  # Stairs down
        17: -1, # Table-tennis
    },
    "KuHar_raw": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17,
    },
    "MotionSense": {
        0: 4,  
        1: 3, 
        2: 0, 
        3: 1, 
        4: 2, 
        5: 5
    },
    "MotionSense_raw": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "WISDM": {
        "A": 2,    # Walking
        "B": 5,    # Jogging
        "C": 6,    # Stairs
        "D": 0,    # Sitting
        "E": 1,    # Standing
        "F": -1,   # Typing
        "G": -1,   # Brushing teeth
        "H": -1,   # Eating soup
        "I": -1,   # Eating chips
        "J": -1,   # Eating pasta
        "K": -1,   # Drinking
        "L": -1,   # Eating sandwich
        "M": -1,   # Kicking
        "O": -1,   # Playing catch
        "P": -1,   # Dribbling
        "Q": -1,   # Writing
        "R": -1,   # Clapping
        "S": -1,   # Folding clothes
    },
    "WISDM_raw": {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "O": 13, "P": 14, "Q": 15, "R": 16, "S": 17,
    },
    "UCI": {
        1: 2,    # walk
        2: 3,    # stair up
        3: 4,    # stair down
        4: 0,    # sit
        5: 1,    # stand
        6: -1,   # Laying
        7: -1,   # stand to sit
        8: -1,   # sit to stand
        9: -1,   # sit to lie
        10: -1,  # lie to sit
        11: -1,  # stand to lie
        12: -1,  # lie to stand
    },
    "UCI_raw": {
        1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9, 11: 10, 12: 11,
    },
    "RealWorld": standard_activity_code_realworld_map,
    "RealWorld_raw": { task: i for i, task in enumerate(tasks) },
    "RecodGait_v1": None,
    "RecodGait": None,
    "GaitOpticalInertial": None,
    "UMAFall": {
        # "ADL": 1,
        # "Fall": -1,
        1: 0, # ADL
        -1: 1, # Fall
    },
    "HAPT": {
        1: 2,    # walk
        2: 3,    # stair up
        3: 4,    # stair down
        4: 0,    # sit
        5: 1,    # stand
        6: 8,    # Laying
        7: 9,    # stand to sit
        8: 9,    # sit to stand
        9: 9,    # sit to lie
        10: 9,   # lie to sit
        11: 9,   # stand to lie
        12: 9,   # lie to stand
    },
    "HAPT_different_transitions": {
        1: 2,    # walk
        2: 3,    # stair up
        3: 4,    # stair down
        4: 0,    # sit
        5: 1,    # stand
        6: 8,    # Laying
        7: 9,    # stand to sit
        8: 10,   # sit to stand
        9: 11,   # sit to lie
        10: 12,  # lie to sit
        11: 13,  # stand to lie
        12: 14,  # lie to stand    
    },
    "HAPT_only_transitions": {
        1: -1,    # walk
        2: -1,    # stair up
        3: -1,    # stair down
        4: -1,    # sit
        5: -1,    # stand
        6: -1,    # Laying
        7: 9,    # stand to sit
        8: 10,   # sit to stand
        9: 11,   # sit to lie
        10: 12,  # lie to sit
        11: 13,  # stand to lie
        12: 14,  # lie to stand 
    },
}

columns_to_rename = {
    "KuHar": None,
    "MotionSense": {
        "userAcceleration.x": "accel-x",
        "userAcceleration.y": "accel-y",
        "userAcceleration.z": "accel-z",
        "rotationRate.x": "gyro-x",
        "rotationRate.y": "gyro-y",
        "rotationRate.z": "gyro-z",
    },
    "WISDM": None,
    "UCI": None,
    "RealWorld": None,
    "RecodGait_v1": None,
    "RecodGait_v2": None,
    "GaitOpticalInertial": {
        "acc x": "accel-x",
        "acc y": "accel-y",
        "acc z": "accel-z",
        "gyro x": "gyro-x",
        "gyro y": "gyro-y",
        "gyro z": "gyro-z",
    },
    "UMAFall": {
        "X-Axis": "accel-x",
        "Y-Axis": "accel-y",
        "Z-Axis": "accel-z",
    },
    "HAPT": None,
    "HAPT_different_transitions": None,
    "HAPT_only_transitions": None,
}

feature_columns: Dict[str, List[str]] = {
    "KuHar": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "MotionSense": [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
        "attitude.roll",
        "attitude.pitch",
        "attitude.yaw",
        "gravity.x",
        "gravity.y",
        "gravity.z",
    ],
    "WISDM": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "UCI": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "RealWorld": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "RecodGait_v1": ["accel-x", "accel-y", "accel-z"],
    "RecodGait_v2": ["accel-x", "accel-y", "accel-z"],
    "GaitOpticalInertial": [
        "accel-x",
        "accel-y",
        "accel-z",
        "gyro-x",
        "gyro-y",
        "gyro-z",
    ],
    "UMAFall": ["accel-x", "accel-y", "accel-z"],
    "HAPT": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "HAPT_different_transitions": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
    "HAPT_only_transitions": ["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
}

match_columns: Dict[str, List[str]] = {
    "KuHar": ["user", "serial", "window", "activity code"],
    "MotionSense": ["user", "serial", "window"],
    "WISDM": ["user", "activity code", "window"],
    "UCI": ["user", "serial", "window", "activity code"],
    "RealWorld": ["user", "window", "activity code", "position"],
    "RealWorld_thigh": ["user", "window", "activity code", "position"],
    "RealWorld_upperarm": ["user", "window", "activity code", "position"],
    "RealWorld_waist": ["user", "window", "activity code", "position"],
    "RecodGait_v1": ["user", "index", "session"],
    "RecodGait_v2": ["user", "index", "session"],
    "GaitOpticalInertial": ["user", "index", "session"],
    "UMAFall": ["file_name"],
    "HAPT": ["user", "serial", "window", "activity code"],
    "HAPT_different_transitions": ["user", "serial", "window", "activity code"],
    "HAPT_only_transitions": ["user", "serial", "window", "activity code"],
}

pipelines: Dict[str, Dict[str, Pipeline]] = {
    "KuHar": {
        "raw_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["KuHar"],
                    column_to_diff="accel-start-time",
                    new_column_name="timestamp diff",
                ),
                Windowize(
                    features_to_select=feature_columns["KuHar"],
                    samples_per_window=300,
                    samples_per_overlap=0,
                    groupby_column=column_group["KuHar"],
                ),
                AddStandardActivityCode(standard_activity_code_map["KuHar_raw"]),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["KuHar"],
                    column_to_diff="accel-start-time",
                    new_column_name="timestamp diff",
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["KuHar"],
                    up=2,
                    down=10,
                    groupby_column=column_group["KuHar"],
                ),
                Windowize(
                    features_to_select=feature_columns["KuHar"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["KuHar"],
                ),
                AddStandardActivityCode(standard_activity_code_map["KuHar"]),
            ]
        ),
    },
    "MotionSense": {
        "raw_dataset": Pipeline(
            [
                RenameColumns(columns_map=columns_to_rename["MotionSense"]),
                Windowize(
                    features_to_select=feature_columns["MotionSense"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["MotionSense"],
                ),
                AddStandardActivityCode(standard_activity_code_map["MotionSense_raw"]),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                RenameColumns(columns_map=columns_to_rename["MotionSense"]),
                AddGravityColumn(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    gravity_columns=["gravity.x", "gravity.y", "gravity.z"],
                ),
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["MotionSense"],
                    up=2,
                    down=5,
                    groupby_column=column_group["MotionSense"],
                ),
                Windowize(
                    features_to_select=feature_columns["MotionSense"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["MotionSense"],
                ),
                AddStandardActivityCode(standard_activity_code_map["MotionSense"]),
            ]
        ),
    },
    "WISDM": {
        "raw_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["WISDM"],
                    column_to_diff="timestamp-accel",
                    new_column_name="accel-timestamp-diff",
                ),
                Windowize(
                    features_to_select=feature_columns["WISDM"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["WISDM"],
                ),
                AddStandardActivityCode(standard_activity_code_map["WISDM_raw"]),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["WISDM"],
                    column_to_diff="timestamp-accel",
                    new_column_name="accel-timestamp-diff",
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=20,
                ),
                Windowize(
                    features_to_select=feature_columns["WISDM"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["WISDM"],
                ),
                AddStandardActivityCode(standard_activity_code_map["WISDM"]),
            ]
        ),
    },
    "UCI": {
        "raw_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["UCI"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["UCI"],
                ),
                AddStandardActivityCode(standard_activity_code_map["UCI_raw"]),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["UCI"],
                    up=2,
                    down=5,
                    groupby_column=column_group["UCI"],
                ),
                Windowize(
                    features_to_select=feature_columns["UCI"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["UCI"],
                ),
                AddStandardActivityCode(standard_activity_code_map["UCI"]),
            ]
        ),
    },
    "RealWorld": {
        "raw_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["RealWorld"],
                    column_to_diff="accel-start-time",
                    new_column_name="timestamp diff",
                ),
                Windowize(
                    features_to_select=feature_columns["RealWorld"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["RealWorld"],
                ),
                AddStandardActivityCode(standard_activity_code_map["RealWorld_raw"]),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["RealWorld"],
                    column_to_diff="accel-start-time",
                    new_column_name="timestamp diff",
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["RealWorld"],
                    up=2,
                    down=5,
                    groupby_column=column_group["RealWorld"],
                ),
                Windowize(
                    features_to_select=feature_columns["RealWorld"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["RealWorld"],
                ),
                AddStandardActivityCode(standard_activity_code_map["RealWorld"]),
            ]
        ),
    },
    "RecodGait_v1": {
        "initial_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["RecodGait_v1"],
                    samples_per_window=256,
                    samples_per_overlap=192,
                    groupby_column=column_group["RecodGait_v1"],
                ),
            ]
        ),
        "raw_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["RecodGait_v1"],
                    samples_per_window=120,
                    samples_per_overlap=0,
                    groupby_column=column_group["RecodGait_v1"],
                ),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["RecodGait_v1"],
                    column_to_diff="accel-start-time",
                    new_column_name="timestamp diff",
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=40,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["RecodGait_v1"],
                    up=2,
                    down=4,
                    groupby_column=column_group["RecodGait_v1"],
                ),
                Windowize(
                    features_to_select=feature_columns["RecodGait_v1"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["RecodGait_v1"],
                ),
            ]
        ),
    },
    "RecodGait_v2": {
        "initial_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["RecodGait_v2"],
                    samples_per_window=256,
                    samples_per_overlap=192,
                    groupby_column=column_group["RecodGait_v2"],
                ),
            ]
        ),
        "raw_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["RecodGait_v2"],
                    samples_per_window=120,
                    samples_per_overlap=0,
                    groupby_column=column_group["RecodGait_v2"],
                ),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                CalcTimeDiffMean(
                    groupby_column=column_group["RecodGait_v2"],
                    column_to_diff="accel-start-time",
                    new_column_name="timestamp diff",
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=40,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["RecodGait_v2"],
                    up=2,
                    down=4,
                    groupby_column=column_group["RecodGait_v2"],
                ),
                Windowize(
                    features_to_select=feature_columns["RecodGait_v2"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["RecodGait_v2"],
                ),
            ]
        ),
    },
    "GaitOpticalInertial": {
        "initial_dataset": Pipeline(
            [
                RenameColumns(columns_map=columns_to_rename["GaitOpticalInertial"]),
                Windowize(
                    features_to_select=feature_columns["GaitOpticalInertial"],
                    samples_per_window=640,
                    samples_per_overlap=480,
                    groupby_column=column_group["GaitOpticalInertial"],
                ),
            ]
        ),
        "raw_dataset": Pipeline(
            [
                RenameColumns(columns_map=columns_to_rename["GaitOpticalInertial"]),
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                Windowize(
                    features_to_select=feature_columns["GaitOpticalInertial"],
                    samples_per_window=300,
                    samples_per_overlap=0,
                    groupby_column=column_group["GaitOpticalInertial"],
                ),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                RenameColumns(columns_map=columns_to_rename["GaitOpticalInertial"]),
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=100,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["GaitOpticalInertial"],
                    up=2,
                    down=10,
                    groupby_column=column_group["GaitOpticalInertial"],
                ),
                Windowize(
                    features_to_select=feature_columns["GaitOpticalInertial"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["GaitOpticalInertial"],
                ),
            ]
        ),
    },
    "UMAFall": {
        "raw_dataset": Pipeline(
            [
                RenameColumns(columns_map=columns_to_rename["UMAFall"]),
                Windowize(
                    features_to_select=feature_columns["UMAFall"],
                    samples_per_window=750,
                    samples_per_overlap=0,
                    groupby_column=column_group["UMAFall"],
                ),
                AddStandardActivityCode(standard_activity_code_map["UMAFall"]),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                RenameColumns(columns_map=columns_to_rename["UMAFall"]),
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    fs=50,
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["UMAFall"],
                    up=2,
                    down=5,
                    groupby_column=column_group["UMAFall"],
                ),
                AddSMV(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                ),
                Peak_Windowize(
                    features_to_select=["accel-x", "accel-y", "accel-z", "smv"],
                    samples_per_window=60,
                    groupby_column=column_group["UMAFall"],
                    peak="random",
                    shift=5,
                ),
                AddStandardActivityCode(standard_activity_code_map["UMAFall"]),
            ]
        ),
    },
    "HAPT": {
        "initial_dataset": Pipeline(
            [
                MedianFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
                    kernel_size=3,
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
                    Wn=20,
                    fs=50,
                    btype="low",
                ),
                # ResamplerPoly(
                #     features_to_select=feature_columns["HAPT"],
                #     up=2,
                #     down=5,
                #     groupby_column=column_group["HAPT"],
                # ),
                Windowize(
                    features_to_select=feature_columns["HAPT"],
                    samples_per_window=128,
                    samples_per_overlap=64,
                    groupby_column=column_group["HAPT"],
                ),
                AddStandardActivityCode(standard_activity_code_map["HAPT"]),
            ]
        ),
        "raw_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["HAPT"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["HAPT"],
                ),
                AddStandardActivityCode(standard_activity_code_map["HAPT"]),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    Wn=0.3,
                    fs=50,
                    btype="high",
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["HAPT"],
                    up=2,
                    down=5,
                    groupby_column=column_group["HAPT"],
                ),
                Windowize(
                    features_to_select=feature_columns["HAPT"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["HAPT"],
                ),
                AddStandardActivityCode(standard_activity_code_map["HAPT"]),
            ]
        ),
    },
    "HAPT_different_transitions": {
        "initial_dataset": Pipeline(
            [
                MedianFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
                    kernel_size=3,
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
                    Wn=20,
                    fs=50,
                    btype="low",
                ),
                Windowize(
                    features_to_select=feature_columns["HAPT_different_transitions"],
                    samples_per_window=128,
                    samples_per_overlap=64,
                    groupby_column=column_group["HAPT_different_transitions"],
                ),
                AddStandardActivityCode(standard_activity_code_map["HAPT_different_transitions"]),
            ]
        ),
        "raw_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["HAPT_different_transitions"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["HAPT_different_transitions"],
                ),
                AddStandardActivityCode(standard_activity_code_map["HAPT_different_transitions"]),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    Wn=0.3,
                    fs=50,
                    btype="high",
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["HAPT_different_transitions"],
                    up=2,
                    down=5,
                    groupby_column=column_group["HAPT_different_transitions"],
                ),
                Windowize(
                    features_to_select=feature_columns["HAPT_different_transitions"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["HAPT_different_transitions"],
                ),
                AddStandardActivityCode(standard_activity_code_map["HAPT_different_transitions"]),
            ]
        ),
    },
    "HAPT_only_transitions": {
        "initial_dataset": Pipeline(
            [
                MedianFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
                    kernel_size=3,
                ),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z"],
                    Wn=20,
                    fs=50,
                    btype="low",
                ),
                Windowize(
                    features_to_select=feature_columns["HAPT_only_transitions"],
                    samples_per_window=128,
                    samples_per_overlap=64,
                    groupby_column=column_group["HAPT_only_transitions"],
                ),
                AddStandardActivityCode(standard_activity_code_map["HAPT_only_transitions"]),
            ]
        ),
        "raw_dataset": Pipeline(
            [
                Windowize(
                    features_to_select=feature_columns["HAPT_only_transitions"],
                    samples_per_window=150,
                    samples_per_overlap=0,
                    groupby_column=column_group["HAPT_only_transitions"],
                ),
                AddStandardActivityCode(standard_activity_code_map["HAPT_only_transitions"]),
            ]
        ),
        "standartized_dataset": Pipeline(
            [
                Convert_G_to_Ms2(axis_columns=["accel-x", "accel-y", "accel-z"]),
                ButterworthFilter(
                    axis_columns=["accel-x", "accel-y", "accel-z"],
                    Wn=0.3,
                    fs=50,
                    btype="high",
                ),
                ResamplerPoly(
                    features_to_select=feature_columns["HAPT_only_transitions"],
                    up=2,
                    down=5,
                    groupby_column=column_group["HAPT_only_transitions"],
                ),
                Windowize(
                    features_to_select=feature_columns["HAPT_only_transitions"],
                    samples_per_window=60,
                    samples_per_overlap=0,
                    groupby_column=column_group["HAPT_only_transitions"],
                ),
                AddStandardActivityCode(standard_activity_code_map["HAPT_only_transitions"]),
            ]
        ),
    },
}

# Creating a list of functions to read the datasets
functions: Dict[str, callable] = {
    "KuHar": read_kuhar,
    "MotionSense": read_motionsense,
    "WISDM": read_wisdm,
    "UCI": read_uci,
    "RealWorld": read_realworld,
    "RecodGait_v1": read_recodGaitV1,
    "RecodGait_v2": read_recodGaitV2,
    "GaitOpticalInertial": read_GaitOpticalInertial,
    "UMAFall": read_umafall,
    "HAPT": read_uci,
    "HAPT_different_transitions": read_uci,
    "HAPT_only_transitions": read_uci,
}

dataset_path: Dict[str, str] = {
    "KuHar": "KuHar/1.Raw_time_domian_data",
    "MotionSense": "MotionSense/A_DeviceMotion_data",
    "WISDM": "WISDM/wisdm-dataset/raw/phone",
    "UCI": "UCI/RawData",
    "RealWorld": "RealWorld/realworld2016_dataset",
    "RecodGait_v1": "RecodGait_v1/RecodGait_v1/device_coordinates",
    "RecodGait_v2": "RecodGait/RecodGait v2/raw_data",
    "GaitOpticalInertial": "GaitOpticalInertial/A multi-sensor human gait dataset/raw_data",
    "UMAFall": "UMAFall/UMAFall_Dataset",
    "HAPT": "UCI/RawData",
    "HAPT_different_transitions": "UCI/RawData",
    "HAPT_only_transitions": "UCI/RawData",
}

# Preprocess the datasets

# Generate ../data/har folder
Path("../data/har").mkdir(parents=True, exist_ok=True) if not os.path.exists("../data/har") else None

view_path = {
    "raw_dataset": "unbalanced",
    "standartized_dataset": "standartized_balanced",
    "standartized_dataset_noGravity": "standartized_noGravity_balanced",
    "standartized_noResample_dataset": "standartized_noResample_balanced",
    "standartized_50Hz_dataset": "standartized_50Hz_balanced",
}

# Path to save the datasets to har task
har_path = Path("../data/har")
har_path.mkdir(parents=True, exist_ok=True) if not os.path.exists("../data/har") else None

output_path_unbalanced_raw: object = Path("../data/har/raw_unbalanced")
output_path_unbalanced_standartized: object = Path(
    "../data/har/standartized_unbalanced"
)

output_path_balanced: object = Path("../data/har/raw_balanced")
output_path_balanced_standartized: object = Path("../data/har/standartized_balanced")

output_path_balanced_user: object = Path("../data/har/raw_balanced_user")
output_path_balanced_standartized_user: object = Path(
    "../data/har/standartized_balanced_user"
)

output_path_balanced_user: object = Path("../data/har/raw_balanced_user")
output_path_balanced_standartized_user: object = Path(
    "../data/har/standartized_balanced_user"
)

output_path_balanced_standartized_50Hz: object = Path(
    "../data/har/standartized_50Hz_balanced"
)

output_path_balanced_standartized_50Hz_noGravity: object = Path(
    "../data/har/standartized_50Hz_noGravity_balanced"
)

# Path to save the datasets to har task for the cpc model
output_path_balanced_cpc: object = Path("../data/har/standartized_cpc_balanced")

# Generate ../data/authentication folder
Path("../data/authentication").mkdir(parents=True, exist_ok=True) if not os.path.exists("../data/authentication") else None

# Path to save the datasets to authentications task
output_path_unbalanced_authentications: object = Path(
    "../data/authentication/unbalanced"
)
output_path_balanced_authentications: object = Path(
    "../data/authentication/raw_balanced"
)
output_path_balanced_standartized_authentications: object = Path(
    "../data/authentication/standartized_balanced"
)

# Generate ../data/fall folder
Path("../data/fall").mkdir(parents=True, exist_ok=True) if not os.path.exists("../data/fall") else None

# Path to save the datasets to fall task
output_path_unbalanced_fall: object = Path("../data/fall/unbalanced")
output_path_balanced_fall: object = Path("../data/fall/raw_balanced")
output_path_balanced_standartized_fall: object = Path(
    "../data/fall/standartized_balanced"
)

# Path to save the datasets to transitions task
output_path_unbalanced_transitions: object = Path("../data/transitions/unbalanced")
output_path_balanced_transitions: object = Path("../data/transitions/raw_balanced")
output_path_balanced_standartized_transitions: object = Path(
    "../data/transitions/standartized_balanced"
)
output_path_unbalanced_raw_transitions: object = Path("../data/transitions/raw_unbalanced")
output_path_unbalanced_standartized_transitions: object = Path(
    "../data/transitions/standartized_unbalanced"
)

# Balncers and splitters used to har task
balancer_activity: object = BalanceToMinimumClass(class_column="standard activity code")
balancer_activity_and_user: object = BalanceToMinimumClassAndUser(
    class_column="standard activity code", filter_column="user", random_state=42
)

split_data: object = SplitGuaranteeingAllClassesPerSplit(
    column_to_split="user",
    class_column="standard activity code",
    train_size=0.8,
    random_state=42,
)

split_data_train_val: object = SplitGuaranteeingAllClassesPerSplit(
    column_to_split="user",
    class_column="standard activity code",
    train_size=0.9,
    random_state=42,
)

# Split LeaveOneSubjectOut used to transitions task
split_leave_one_subject_out: object = LeaveOneSubjectOut(column_to_split="user")

def balance_per_activity(
    dataset: str, dataframe: pd.DataFrame, output_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """This function balance the dataset per activity and save the balanced dataset.

    Parameters
    ----------
    dataset : str
        The dataset name.
    dataframe : pd.DataFrame
        The dataset.
    output_path : str
        The path to save the balanced dataset.

    Returns
    -------
    None
    """

    random.seed(42)
    np.random.seed(42)

    train_df, test_df = split_data(dataframe)
    train_df, val_df = split_data_train_val(train_df)

    train_df = balancer_activity(train_df)
    val_df = balancer_activity(val_df)
    test_df = balancer_activity(test_df)

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

def split_per_user(dataset, dataframe, output_path):
    """The function balance the dataset per user and save the balanced dataset.

    Parameters
    ----------
    dataset : str
        The dataset name.
    dataframe : pd.DataFrame
        The dataset.
    output_path : str
        The path to save the balanced dataset.

    Returns
    -------
    None
    """

    random.seed(42)
    np.random.seed(42)

    train_df, test_df = split_data(dataframe)
    train_df, val_df = split_data_train_val(train_df)

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

def balance_per_user_and_activity(dataset, dataframe, output_path):
    """The function balance the dataset per user and activity and save the balanced dataset.

    Parameters
    ----------
    dataset : str
        The dataset name.
    dataframe : pd.DataFrame
        The dataset.
    output_path : str
        The path to save the balanced dataset.

    Returns
    -------
    None
    """

    random.seed(42)
    np.random.seed(42)

    train_df, test_df = split_data(dataframe)
    train_df, val_df = split_data_train_val(train_df)

    train_df = balancer_activity_and_user(train_df)
    val_df = balancer_activity_and_user(val_df)
    test_df = balancer_activity_and_user(test_df)

    # new_df_balanced = balancer_activity_and_user(
    #     dataframe[dataframe["standard activity code"] != -1]
    # )

    output_dir = output_path / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

def generate_views(new_df, new_df_standartized, dataset):
    """This function generate the views of the dataset.

    Parameters
    ----------
    new_df : pd.DataFrame
        The raw dataset.
    new_df_standartized : pd.DataFrame
        The standartized dataset.
    dataset : str
        The dataset name.
    """

    random.seed(42)
    np.random.seed(42)

    # Preprocess and save the raw dataset unbalanced per activity
    split_per_user(dataset, new_df, output_path_unbalanced_raw)

    # Preprocess and save the standartized dataset unbalanced per activity
    split_per_user(dataset, new_df_standartized, output_path_unbalanced_standartized)

    # # Filter the datasets by equal elements
    
    filter_common = FilterByCommonRows(match_columns=match_columns[dataset])
    new_df, new_df_standartized = filter_common(new_df, new_df_standartized)

    # Preprocess and save the raw balanced dataset per user and activity
    balance_per_user_and_activity(
        dataset, new_df, output_path_balanced_user
    )

    # Preprocess and save the raw balanced dataset per activity
    balance_per_activity(
        dataset, new_df, output_path_balanced
    )

    # Preprocess and save the standartized balanced dataset per user and activity
    balance_per_user_and_activity(
        dataset, new_df_standartized, output_path_balanced_standartized_user
    )

    # Preprocess and save the standartized balanced dataset per activity
    balance_per_activity(
        dataset, new_df_standartized, output_path_balanced_standartized
    )


############################################################################################################################
# This part of the code has the functions to balance and generate the dataset views to fall task
############################################################################################################################

def generate_fall_initial_view(df, output_dir, n_folds: int = 5):
    """This function generate the initial view of the dataset to fall task and save the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The raw dataset.
    output_dir : str
        The path to save the balanced dataset.
    n_folds : int, optional
        The number of folds, by default 5

    Returns
    -------
    None
    """

    random.seed(42)
    np.random.seed(42)    
    
    df['user'] = df['user'].astype(int)
    users = df["user"].unique()
    # Order the users by descending number
    users = np.sort(users)
    random.seed(42)
    seed = 42

    # Select the users without Fall samples
    users_without_fall = []
    for (user, df_user) in df.groupby("user"):
        # if "Fall" not in df_user["activity code"].unique():
        if -1 not in df_user["activity code"].unique():
            users_without_fall.append(user)

    print(f"Users without Fall: {users_without_fall}")

    users_with_fall = [user for user in users if user not in users_without_fall]
    print(f"Users with Fall: {users_with_fall}")
    random.shuffle(users_with_fall)
    print(f"Users with Fall shuffled: {users_with_fall}")

    total_users_fall = len(users_with_fall)
    print(f"Total users with Fall: {total_users_fall}")
    print(f"Total users without Fall: {len(users_without_fall)}")

    total_users = len(users)
    percent = 0.5

    num_users_with_fall = np.ceil(total_users * percent).astype(int) 
    users_test = users_with_fall[:num_users_with_fall]

    # Add the users without Fall samples in the train
    users_train = users_with_fall[num_users_with_fall:] + users_without_fall

    # print(f"Users in the train: {sorted(users_train)}")
    # print(f"Users in the test: {sorted(users_test)}")

    df_train = df[df["user"].isin(users_train)].copy()
    df_test = df[df["user"].isin(users_test)].copy()

    # Remove the samples with Fall from the train
    # df_train = df_train[df_train["activity code"] != "Fall"].copy()
    df_train = df_train[df_train["activity code"] != -1].copy()

    k_folds = 5
    n_test_users = len(users_test)

    users_per_fold = int(n_test_users / k_folds)

    # We will use the selected users to train and separete the users to test in n_folds folds
    for i in range(k_folds):
        test_users = users_test[i * users_per_fold : (i + 1) * users_per_fold]

        df_test_fold = df_test[df_test["user"].isin(test_users)].copy()
        df_train_fold = df_train.copy()

        df_train_fold.reset_index(drop=True, inplace=True)
        df_test_fold.reset_index(drop=True, inplace=True)

        # Let's save the train and test
        output_dir_fold = output_dir / f"fold_{i}"
        output_dir_fold.mkdir(parents=True, exist_ok=True) if not os.path.exists(
            output_dir_fold
        ) else None
        df_train_fold.to_csv(output_dir_fold / "train.csv", index=False)
        df_test_fold.to_csv(output_dir_fold / "test.csv", index=False)

def generate_fall_views(new_df, new_df_standartized, dataset):
    """This function generate the views of the dataset.

    Parameters
    ----------
    new_df : pd.DataFrame
        The raw dataset.
    new_df_standartized : pd.DataFrame
        The standartized dataset.
    dataset : str
        The dataset name.
    """
    random.seed(42)
    np.random.seed(42)

    # Save the unbalanced dataset
    output_dir = output_path_unbalanced_fall / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess and save the raw dataset
    output_dir = output_path_balanced_fall / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_dir / "unbalanced.csv", index=False)

    # Preprocess and save the raw dataset
    output_dir = output_path_balanced_fall / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_fall_initial_view(new_df, output_dir)

    # Preprocess and save the standartized dataset
    output_dir = output_path_balanced_standartized_fall / dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_fall_initial_view(new_df_standartized, output_dir)


############################################################################################################################
# This part of the code has the functions to balance and generate the dataset views to authentication task
############################################################################################################################
def balance_authentication(dataset, dataframe, output_path_balanced, n_folds=5):
    # Let's take the second minimum value of samples per user
    users = dataframe["user"].unique()

    random.seed(42)
    np.random.seed(42)
    
    # Objects to generate the folds
    folder_positive: object = GenerateFold(
        column_to_split="session acc", n_folds=2, random_state=42
    )
    folder_negatives: object = GenerateFold(
        column_to_split="user", n_folds=5, random_state=42
    )

    for user in users:
        user_df = dataframe.copy()
        user_df["authentication"] = user_df["user"].apply(
            lambda x: 1 if x == user else 0
        )

        negative_user_df = user_df[user_df["authentication"] == 0].copy()

        positive_user_df = user_df[user_df["authentication"] == 1].copy()
        positive_user_df_balanced = folder_positive(positive_user_df)

        # Let's split the folds in train and test
        for i in range(n_folds):
            train, test = None, None

            negative_user_folders = folder_negatives(negative_user_df)

            for j in range(n_folds):
                if i == j:
                    num_users = len(negative_user_folders[i]["user"].unique())
                    min_samples_per_user = int(
                        positive_user_df_balanced[1].shape[0] / num_users
                    )

                    # Objects to balance the dataset
                    balancer_samples: object = BalanceToMinimumClass(
                        class_column="user",
                        random_state=42,
                        min_value=min_samples_per_user,
                    )
                    negative_user_folders[i] = balancer_samples(
                        negative_user_folders[i]
                    )

                    test = pd.concat(
                        [positive_user_df_balanced[1], negative_user_folders[i]]
                    )

                else:
                    num_users = len(negative_user_folders[j]["user"].unique())
                    min_samples_per_user = int(
                        positive_user_df_balanced[0].shape[0]
                        / (num_users * (n_folds - 1))
                    )

                    # Objects to balance the dataset
                    balancer_samples: object = BalanceToMinimumClass(
                        class_column="user",
                        random_state=42,
                        min_value=min_samples_per_user,
                    )
                    negative_user_folders[j] = balancer_samples(
                        negative_user_folders[j]
                    )
                    if train is None:
                        train = pd.concat(
                            [positive_user_df_balanced[0], negative_user_folders[j]]
                        )
                    else:
                        train = pd.concat([train, negative_user_folders[j]])

            test.reset_index(drop=True, inplace=True)
            train.reset_index(drop=True, inplace=True)

            # Let's save the train and test
            output_dir = output_path_balanced / dataset / str(user) / f"fold_{i}"
            output_dir.mkdir(parents=True, exist_ok=True) if not os.path.exists(
                output_dir
            ) else None
            train.to_csv(output_dir / "train.csv", index=False)
            test.to_csv(output_dir / "test.csv", index=False)


def generate_authentication_initial_view(new_df_initial, output_dir):
    """This function generate the initial view of the dataset to authentication task.

    Parameters
    ----------
    new_df_initial : pd.DataFrame
        The raw dataset.
    output_dir : str
        The path to save the balanced dataset.

    Returns
    -------
    None
    """

    random.seed(42)
    np.random.seed(42)

    users = new_df_initial["user"].unique()
    sessions = new_df_initial["session"].unique()

    min_samples_per_user = new_df_initial.groupby(["user", "session"]).size().min()
    min_value = min(min_samples_per_user, 75)
    balancer_session = BalanceToMinimumClass(
        class_column="session", random_state=42, min_value=min_value
    )

    for user in users:
        user_df = new_df_initial.copy()
        user_df["authentication"] = user_df["user"].apply(
            lambda x: 1 if x == user else 0
        )

        positive_user_df = user_df[user_df["authentication"] == 1].copy()
        negative_user_df = user_df[user_df["authentication"] == 0].copy()

        negative_users = list(negative_user_df["user"].unique())

        random.seed(42)
        random.shuffle(negative_users)
        num_negative_users = len(negative_users) // 2

        train_negative_users = negative_user_df[
            negative_user_df["user"].isin(negative_users[:num_negative_users])
        ].copy()
        test_negative_users = negative_user_df[
            negative_user_df["user"].isin(negative_users[num_negative_users:])
        ].copy()

        # Let's select 75 aleatory samples
        train_negative_users = train_negative_users.sample(n=75, random_state=42)
        val_negative_users = train_negative_users.sample(n=15, random_state=42) # Take 15 samples to validation (20 %)
        train_negative_users = train_negative_users.drop(val_negative_users.index) # Remove the samples from the train
        test_negative_users = test_negative_users.sample(n=75, random_state=42)

        output_dir_user = output_dir / str(user)
        output_dir_user.mkdir(parents=True, exist_ok=True) if not os.path.exists(
            output_dir_user
        ) else None

        positive_user_df = balancer_session(positive_user_df)

        train_positive_user = positive_user_df[positive_user_df["session"] == sessions[0]]
        total = train_positive_user.shape[0]
        val_positive_user = positive_user_df.sample(n=int(total * 0.2), random_state=42)
        train_positive_user = positive_user_df.drop(val_positive_user.index)

        train = pd.concat(
            [
                # positive_user_df[positive_user_df["session"] == sessions[0]],
                train_positive_user,
                train_negative_users,
            ]
        )
        validation = pd.concat(
            [
                # positive_user_df[positive_user_df["session"] == sessions[1]],
                val_positive_user,
                val_negative_users,
            ]
        )

        test = pd.concat(
            [
                positive_user_df[positive_user_df["session"] == sessions[1]],
                test_negative_users,
            ]
        )

        train.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)

        # Let's save the train and test
        train.to_csv(output_dir_user / "train.csv", index=False)
        validation.to_csv(output_dir_user / "validation.csv", index=False)
        test.to_csv(output_dir_user / "test.csv", index=False)


def generate_authentication_views(new_df, new_df_standartized, dataset, output_dir):
    """This function generate the views of the dataset to authentication task.

    Parameters
    ----------
    new_df : pd.DataFrame
        The raw dataset.
    new_df_standartized : pd.DataFrame
        The standartized dataset.
    dataset : str
        The dataset name.
    output_dir : str
        The path to save the balanced dataset.

    Returns
    -------
    None
    """

    random.seed(42)
    np.random.seed(42)

    # Filter the datasets by equal elements
    filter_common = FilterByCommonRows(match_columns=match_columns[dataset])
    new_df, new_df_standartized = filter_common(new_df, new_df_standartized)

    # Save the raw balanced dataset
    output_dir_raw = output_dir / "raw_balanced" / dataset
    output_dir_raw.mkdir(parents=True, exist_ok=True) if not os.path.exists(
        output_dir_raw
    ) else None
    generate_authentication_initial_view(new_df, output_dir_raw)

    # Save the standartized balanced dataset
    output_dir_standartized = output_dir / "standartized_balanced" / dataset
    output_dir_standartized.mkdir(parents=True, exist_ok=True) if not os.path.exists(
        output_dir_standartized
    ) else None
    generate_authentication_initial_view(new_df_standartized, output_dir_standartized)

def generate_cpc_view(cpc_dataset, dataset):

    # Split the dataset in train, validation and test
    train, test = split_data(cpc_dataset)
    train, validation = split_data_train_val(train)

    for user, df in train.groupby("user"):
        output_dir = output_path_balanced_cpc / dataset / "train" / f"{user}.csv"
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(output_dir, index=False)

    for user, df in validation.groupby("user"):
        output_dir = output_path_balanced_cpc / dataset / "validation" / f"{user}.csv"
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(output_dir, index=False)

    for user, df in test.groupby("user"):
        output_dir = output_path_balanced_cpc / dataset / "test" / f"{user}.csv"
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(output_dir, index=False)

############################################################################################################################
# This part of the code is used to generate the datasets views for the har task
############################################################################################################################

# Creating the datasets views fot the har task
# for dataset in har_datasets:

#     print(f"Preprocessing the {dataset} dataset ...\n")
#     os.mkdir(output_path_unbalanced_raw) if not os.path.isdir(output_path_unbalanced_raw) else None
#     os.mkdir(output_path_unbalanced_standartized) if not os.path.isdir(output_path_unbalanced_standartized) else None
#     os.mkdir(output_path_balanced) if not os.path.isdir(output_path_balanced) else None
#     os.mkdir(output_path_balanced_standartized) if not os.path.isdir(output_path_balanced_standartized) else None
#     os.mkdir(output_path_balanced_user) if not os.path.isdir(output_path_balanced_user) else None
#     os.mkdir(output_path_balanced_standartized_user) if not os.path.isdir(output_path_balanced_standartized_user) else None
#     os.mkdir(output_path_balanced_cpc) if not os.path.isdir(output_path_balanced_cpc) else None

#     reader = functions[dataset]

#     # Verify if the file unbalanced.csv is already created

#     # Read the raw dataset
#     if dataset == "RealWorld":
#         rw_thigh = "RealWorld_thigh"
#         rw_waist = "RealWorld_waist"

#         print("Organizing the RealWorld dataset ...\n")
#         # Create a folder to save the organized dataset
#         workspace = Path(
#             "../data/original/RealWorld/realworld2016_dataset_organized"
#         )

#         os.mkdir(workspace) if not os.path.isdir(workspace) else None

#         # Organize the dataset
#         workspace, users = real_world_organize()
#         path = workspace
#         raw_dataset = reader(path, users)
#         # Preprocess the raw dataset
#         new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
#         # Preprocess the standartized dataset
#         new_df_standartized = pipelines[dataset]["standartized_dataset"](
#             raw_dataset
#         )

#         # Remove activities that are equal to -1
#         # new_df = new_df[new_df["standard activity code"] != -1]
#         new_df_standartized = new_df_standartized[
#             new_df_standartized["standard activity code"] != -1
#         ]

#         generate_views(new_df, new_df_standartized, dataset)
        
#         positions = ["thigh", "waist"]
#         for position in list(positions):
#             print(f"Generating the views for the {dataset} dataset for the {position} position\n")
#             new_df_filtered = new_df[new_df["position"] == position]
#             new_df_standartized_filtered = new_df_standartized[
#                 new_df_standartized["position"] == position
#             ]
#             new_dataset = dataset + "_" + position

#             generate_views(
#                 new_df_filtered, new_df_standartized_filtered, new_dataset
#             )
            
#     else:
#         path = f"../data/original/{dataset_path[dataset]}"
#         raw_dataset = reader(path)
#         print(f"Total of activities in entire dataset: {raw_dataset['activity code'].nunique()}")

#         # Preprocess the raw dataset
#         new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
#         print(f"Total of activities in entire dataset after the pipeline: {new_df['standard activity code'].nunique()}")

#         # Preprocess the standartized dataset
#         new_df_standartized = pipelines[dataset]["standartized_dataset"](
#             raw_dataset
#         )
#         # Remove activities that are equal to -1
#         # new_df = new_df[new_df["standard activity code"] != -1]
#         new_df_standartized = new_df_standartized[
#             new_df_standartized["standard activity code"] != -1
#         ]
#         generate_views(new_df, new_df_standartized, dataset)

############################################################################################################################
# Creating the datasets views fot the authentication task
############################################################################################################################
for dataset in authentications_datasets:
    # Verify if the dataset is already created
    if os.path.isdir(output_path_unbalanced_authentications / dataset):
        print(f"The dataset {dataset} already was created.\n")

    else:
        print(f"Preprocessing the {dataset} dataset ...\n")

        reader = functions[dataset]

        # Read the raw dataset
        path = Path(f"../data/original/{dataset_path[dataset]}")
        raw_dataset = reader(path)

        # Preprocess the initial dataset
        path = Path(f"../data/authentication")
        os.mkdir(path) if not os.path.isdir(path) else None
        path = Path(f"../data/authentication/initial_dataset")
        os.mkdir(path) if not os.path.isdir(path) else None
        path = Path(f"../data/authentication/initial_dataset/{dataset}")
        os.mkdir(path) if not os.path.isdir(path) else None

        new_df_initial = pipelines[dataset]["initial_dataset"](raw_dataset)
        print(f"Generating the initial dataset views for {dataset} ...\n")
        generate_authentication_initial_view(new_df_initial, path)

        # Create folders to save the unbalanced dataset
        os.mkdir(output_path_unbalanced_authentications) if not os.path.isdir(output_path_unbalanced_authentications) else None
        os.mkdir(output_path_unbalanced_authentications / dataset) if not os.path.isdir(output_path_unbalanced_authentications / dataset) else None

        # Create folders to save the balanced dataset
        os.mkdir(output_path_balanced_authentications) if not os.path.isdir(output_path_balanced_authentications) else None
        os.mkdir(output_path_balanced_authentications / dataset) if not os.path.isdir(output_path_balanced_authentications / dataset) else None
        os.mkdir(output_path_balanced_standartized_authentications) if not os.path.isdir(output_path_balanced_standartized_authentications) else None
        os.mkdir(output_path_balanced_standartized_authentications / dataset) if not os.path.isdir(output_path_balanced_standartized_authentications / dataset) else None

        # Preprocess the raw dataset
        new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
        new_df.to_csv(
            output_path_unbalanced_authentications / dataset / "raw_unbalanced.csv",
            index=False,
        )

        # Preprocess the standartized dataset
        new_df_standartized = pipelines[dataset]["standartized_dataset"](raw_dataset)
        new_df_standartized.to_csv(
            output_path_unbalanced_authentications
            / dataset
            / "standartized_unbalanced.csv",
            index=False,
        )

        # Generate the balanced dataset
        output_dir = Path("../data/authentication")
        generate_authentication_views(new_df, new_df_standartized, dataset, output_dir)

# Creating the datasets views fot the fall task
for dataset in fall_datasets:
    # Verify if the dataset is already created
    if os.path.isdir(output_path_unbalanced_fall / dataset):
        print(f"The dataset {dataset} already was created.\n")

    else:
        print(f"Preprocessing the {dataset} dataset ...\n")

        reader = functions[dataset]

        # Read the raw dataset
        path = Path(f"../data/original/{dataset_path[dataset]}")
        raw_dataset = reader(path)

        # Preprocess the raw dataset
        new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
        # Preprocess the standartized dataset
        new_df_standartized = pipelines[dataset]["standartized_dataset"](
            raw_dataset
        )
        print(new_df_standartized['standard activity code'].unique())
        generate_fall_views(new_df, new_df_standartized, dataset)

# Creating the datasets views fot the transitions task
for dataset in transitions_datasets:
    # Verify if the dataset is already created
    if os.path.isdir(output_path_unbalanced_transitions / dataset):
        print(f"The dataset {dataset} already was created.\n")
    else:
        print(f"Preprocessing the {dataset} dataset ...\n")

        reader = functions[dataset]

        # Read the raw dataset
        path = Path(f"../data/original/{dataset_path[dataset]}")
        raw_dataset = reader(path)
        raw_dataset['user'] = raw_dataset['user'].astype(int)

        path = Path(f"../data/transitions")
        os.mkdir(path) if not os.path.isdir(path) else None

        # Preprocess the initial dataset
        unbalanced_path = Path(f"../data/transitions/initial_dataset_unbalanced")
        os.mkdir(unbalanced_path) if not os.path.isdir(unbalanced_path) else None
        unbalanced_path = unbalanced_path / dataset
        os.mkdir(unbalanced_path) if not os.path.isdir(unbalanced_path) else None

        balanced_path = Path(f"../data/transitions/initial_dataset_balanced")
        os.mkdir(balanced_path) if not os.path.isdir(balanced_path) else None
        balanced_path = balanced_path / dataset
        os.mkdir(balanced_path) if not os.path.isdir(balanced_path) else None

        print(f"Generating the initial dataset views for {dataset} ...\n")
        new_df_initial = pipelines[dataset]["initial_dataset"](raw_dataset)
        new_df_initial[new_df_initial["standard activity code"] != -1]

        dfs = split_leave_one_subject_out(new_df_initial)
        for split in dfs.keys():
            output_dir_fold_unbalanced = unbalanced_path / str(split)
            output_dir_fold_unbalanced.mkdir(parents=True, exist_ok=True)
            output_dir_fold_balanced = balanced_path / str(split)
            output_dir_fold_balanced.mkdir(parents=True, exist_ok=True)

            train, test = dfs[split]
            train.to_csv(output_dir_fold_unbalanced / "train.csv", index=False)
            test.to_csv(output_dir_fold_unbalanced / "test.csv", index=False)

            train = balancer_activity(train)
            test = balancer_activity(test)
            train.to_csv(output_dir_fold_balanced / "train.csv", index=False)
            test.to_csv(output_dir_fold_balanced / "test.csv", index=False)

        # Preprocess the raw dataset
        new_df = pipelines[dataset]["raw_dataset"](raw_dataset)
        new_df_initial = new_df_initial[new_df_initial["standard activity code"] != -1]
        
        # Preprocess the standartized dataset
        new_df_standartized = pipelines[dataset]["standartized_dataset"](
            raw_dataset
        )
        new_df_standartized = new_df_standartized[
            new_df_standartized["standard activity code"] != -1
        ]

        # Save the unbalanced dataset
        output_dir = output_path_unbalanced_transitions / dataset
        output_dir.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(output_dir / "raw_unbalanced.csv", index=False)
        new_df_standartized.to_csv(output_dir / "standartized_unbalanced.csv", index=False)

        for (view, df) in [('raw', new_df), ('standartized', new_df_standartized)]:
            dfs = split_leave_one_subject_out(df)
            for split in dfs.keys():
                if view == 'raw':
                    output_dir_unbalanced = output_path_unbalanced_raw_transitions / dataset
                    output_dir_balanced = output_path_balanced_transitions / dataset
                else:
                    output_dir_unbalanced = output_path_unbalanced_standartized_transitions / dataset
                    output_dir_balanced = output_path_balanced_standartized_transitions / dataset
                output_dir_unbalanced.mkdir(parents=True, exist_ok=True)
                output_dir_balanced.mkdir(parents=True, exist_ok=True)

                output_dir_unbalanced_fold = output_dir_unbalanced / str(split)
                output_dir_unbalanced_fold.mkdir(parents=True, exist_ok=True)
                output_dir_balanced_fold = output_dir_balanced / str(split)
                output_dir_balanced_fold.mkdir(parents=True, exist_ok=True)

                # train, test = dfs[split]
                train, val, test = split_data_train_val(dfs[split])
                train.to_csv(output_dir_unbalanced_fold / "train.csv", index=False)
                val.to_csv(output_dir_unbalanced_fold / "validation.csv", index=False)
                test.to_csv(output_dir_unbalanced_fold / "test.csv", index=False)

                # Generate the balanced dataset
                train = balancer_activity(train)
                test = balancer_activity(test)
                train.to_csv(output_dir_balanced_fold / "train.csv", index=False)
                test.to_csv(output_dir_balanced_fold / "test.csv", index=False)

# Remove the junk folder
workspace = Path("../data/processed")
if os.path.isdir(workspace):
    shutil.rmtree(workspace)

# Remove the realworld2016_dataset_organized folder
workspace = Path("../data/original/RealWorld/realworld2016_dataset_organized")
if os.path.isdir(workspace):
    shutil.rmtree(workspace)
