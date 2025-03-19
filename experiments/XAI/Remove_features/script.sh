python3 remove_best_feature.py -d ../experiments/UMAP_axis/results/execution/transformed_data/ -o UMAP_axis_results_aux/feature_importance_remove_best/ -r axis

python3 remove_worst_feature.py -d ../experiments/UMAP_axis/results/execution/transformed_data/ -o UMAP_axis_results_aux/feature_importance_remove_worst/ -r axis

python3 remove_best_feature_oracle.py -d ../experiments/UMAP_axis/results/execution/transformed_data/ -o UMAP_axis_results_aux/feature_importance_remove_best_oracle/

python3 remove_worst_feature_oracle.py -d ../experiments/UMAP_axis/results/execution/transformed_data/ -o UMAP_axis_results_aux/feature_importance_remove_worst_oracle/
