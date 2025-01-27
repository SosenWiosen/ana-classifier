import os
import json
import tensorflow as tf

def load_models(models_folder):
    print("Loading TensorFlow exported models, base directory:", models_folder)
    models = {}
    print(os.listdir(models_folder))
    for model_dir in os.listdir(models_folder):
        full_model_dir = os.path.join(models_folder, model_dir)
        model_config_path = os.path.join(full_model_dir, 'model_config.json')

        if os.path.isdir(full_model_dir) and os.path.isfile(model_config_path):
            with open(model_config_path, 'r') as f:
                config = json.load(f)

            model_path = os.path.join(full_model_dir, config['model_path'])
            print(model_path)
            labels = config['labels']
            copy_red_channel = config.get('copy_red_channel', False)
            try:
                loaded_model = tf.saved_model.load(model_path)
                models[config['name']] = {
                    'config': config,
                    'loaded_model': loaded_model,
                    'labels': labels,
                    'copy_red_channel': copy_red_channel
                }
            except Exception as e:
                print(f"Failed to load model {config['name']}. Error: {e}")
    print(models)
    return models
