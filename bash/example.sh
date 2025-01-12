#!/bin/bash


method="GA"
ablation_study="all"
file_dir="your/file/to/save/generated/alphas/and/code"
fitness_method="Pearson_IC"
selection_strategy="roulette"
update_strategy="cut"
data_type="train"
generate_model_name="gpt-3.5-turbo-1106"
max_generations=10
population_size=10

echo "Method: $method"
echo "Ablation Study: $ablation_study"
echo "File Directory: $file_dir"
echo "Fitness Method: $fitness_method"
echo "Selection Strategy: $selection_strategy"
echo "Update Strategy: $update_strategy"
echo "Data Type: $data_type"
echo "Generate Model Name": $generate_model_name
echo "Max Generations": max_generations
echo "Population Size": population_size


python ./EvolutionaryAlgorithmAgent.py --method "$method" --ablation_study "$ablation_study" --file_dir "$file_dir" --fitness_method "$fitness_method" --selection_strategy "$selection_strategy" --update_strategy "$update_strategy" --data_type "$data_type" --generate_model_name "$generate_model_name" --max_generations "$max_generations" --population_size "$population_size"
