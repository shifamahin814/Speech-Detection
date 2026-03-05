import os
import subprocess

# Define model-dataset combinations
combinations = [
    ('model1', 'dataset1'),
    ('model1', 'dataset2'),
    ('model2', 'dataset1'),
    ('model2', 'dataset2'),
    # Add more combinations as needed
]

# Train all models on all datasets
for model, dataset in combinations:
    print(f'Training {model} on {dataset}...')
    subprocess.run(['python', f'train_{model}.py', '--dataset', dataset])
    print(f'Done training {model} on {dataset}.')
