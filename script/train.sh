#!/bin/bash

# Define the directory names
directories=( "bike" "buu"   )

# Loop through each directory name
for dir_name in "${directories[@]}"; do


    python3 train.py --source_path LOMlightdiffusion/"$dir_name"/ --model_path LOMlightdiffusion/"$dir_name"/output --eval -r 1
    python render.py -m LOMlightdiffusion/"$dir_name"/output
    python metrics.py -m LOMlightdiffusion/"$dir_name"/output
done

