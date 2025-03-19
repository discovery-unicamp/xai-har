# How to reproduce the experiments?

The experiments can be executed in different domains: time, frequency, dimensionality reduction.

To execute we need the data in the desired domain and is requested that the folder that contain the dataset has the following name: `<name_dst_train>_<name_dst_test>_<domain>`.

## UMAP

```
python remove_worst_feature.py -d 
```

```
python remove_best_feature.py -d 
```

## KPCA

```
python remove_worst_feature.py -d 
```

```
python remove_best_feature.py -d 
```

## Frequency

```
python remove_worst_feature.py -d ../experiments/Frequency/results/execution/transformed_data/ -o Frequency_results/feature_importance_remove_worst/ -r freq
```

```
python remove_best_feature.py -d ../experiments/Frequency/results/execution/transformed_data/ -o Frequency_results/feature_importance_remove_best/ -r freq
```

## Time

