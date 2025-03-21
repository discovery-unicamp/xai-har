#!/bin/bash
# Author: Patrick Alves
# email: tricksantos88@gmail.com

# Install all necessary dependencies
pip install -r requirements.txt

# Generate the inter standardized views
cd generator
python3 inter_dataset_balancer.py


# This command will execute the har experiments
echo "\nExecuting the har experiments"
cd ../experiments/experiment_executor
python3 create_configs_har.py
# python3 execute.py har_experiments/experiments/ -d ../../data/ -o har_experiments/results --skip-existing --ray