#!/bin/bash
# Author: Patrick Alves
# email: tricksantos88@gmail.com

# This command will execute the har experiments
echo "\nExecuting the har experiments"
python3 create_configs_har.py
python3 execute.py har_experiments/experiments/ -d ../../data/ -o har_experiments/results --skip-existing --ray