import json
import os

import tensorflow as tf


def load_models(models_folder):
    models = {}
    for model_dir in os.listdir(models_folder):
        full_model_dir = os.path.join(models_folder, model_dir)
        model_config_path = os.path.join(full_model_dir, 'model_config.json')

        if os.path.isdir(full_model_dir) and os.path.isfile(model_config_path):
            with open(model_config_path, 'r') as f:
                config = json.load(f)

            model_path = os.path.join(full_model_dir, config['model_path'])
            labels = config['labels']
            copy_red_channel = config.get('copy_red_channel', False)
            try:
                models[config['name']] = {
                    'config': config,
                    'labels': labels,
                    'model_path': model_path,
                    'copy_red_channel': copy_red_channel
                }
            except Exception as e:
                print(f"Failed to load model {config['name']}. Error: {e}")
    return models
