from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import uuid
import jwt
import json
import datetime
from functools import wraps
import numpy as np
import tensorflow as tf
import base64  # For handling base64 encoding
from PIL import Image  # For image manipulation
from io import BytesIO  # For handling image as a byte buffer

app = Flask(__name__)
CORS(app)
# Configuration
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MODELS_FOLDER'] = "./models/"


db = SQLAlchemy(app)

# Database Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

class RequestLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_name = db.Column(db.String(100), nullable=False)
    prediction_result = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)


models = {}  # This will store model instances

def load_models():
    models_folder = app.config['MODELS_FOLDER']
    print("Loading TensorFlow exported models, base directory:", models_folder)
    for model_dir in os.listdir(models_folder):
        print("Found directory:", model_dir)
        saved_model_dir = os.path.join(models_folder, model_dir)  # Path to the SavedModel directory
        model_config_path = os.path.join(models_folder, model_dir, 'model_config.json')

        if os.path.isdir(saved_model_dir) and os.path.isfile(model_config_path):
            print("Found model directory with config:", saved_model_dir)

            # Load the model configuration
            with open(model_config_path, 'r') as f:
                config = json.load(f)

            model_name = config['name']  # Retrieve the model's name
            model_path = config['model_path']  # Retrieve the model's path
            try:
                # Join the model path with the SavedModel directory
                model_file = os.path.join(saved_model_dir, model_path)
                # Load the TensorFlow SavedModel
                loaded_model = tf.saved_model.load(model_file)
                print(f"Successfully loaded TensorFlow model: {model_name} from {saved_model_dir}")

                # Store the loaded model along with its config into the models dictionary
                models[model_name] = {
                    'model_path': model_file,  # Store the directory of the SavedModel
                    'config': config,
                    'loaded_model': loaded_model
                }
            except Exception as e:
                print(f"Failed to load model {model_name} from {model_file}. Error: {e}")
        else:
            print(f"No valid config or SavedModel found in directory: {model_dir}")

    print("Models loaded:", list(models.keys()))

def decode_and_preprocess_image(image_data, model_config):
    """
    Preprocess image data to match the training preprocessing pipeline.

    Args:
        image_data (str): The base64-encoded string of the image.
        model_config (dict): A dictionary containing model attributes, such as `input_shape`

    Returns:
        tf.Tensor: A preprocessed image tensor, ready for inference.
    """
    # Decode base64 image data and load it as a PIL image
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    image = image.convert("RGB")  # Ensure 3 channels (RGB)

    # Resize image to match model input size
    target_size = (model_config["input_shape"][0], model_config["input_shape"][1])
    image = image.resize(target_size)

    # Convert to TensorFlow tensor and normalize to [0, 1]
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)  # TensorFlow tensor
    # Add batch dimension (TensorFlow expects a batch of inputs for inference)
    image_tensor = tf.expand_dims(image_tensor, axis=0)  # Shape: (1, height, width, 3)
    if model_config["copy_red_channel"]:
        red_channel = image_tensor[..., 0:1]  # Extract only the red channel, shape (H, W, 1)
        image_tensor = tf.concat([red_channel, red_channel, red_channel], axis=-1)
    return image_tensor


# def get_model(model_name):
#     model_info = models.get(model_name)
#     if model_info and model_info['loaded_model'] is None:
#         # Load and cache the model
#         model_info['loaded_model'] =
#         print(f"Loaded model {model_name} from {model_info['model_path']}")
#     return model_info['loaded_model'] if model_info else None

load_models()

# Decorator for authorization
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('x-access-token')
        if not token:
            return jsonify({'message': 'Token is missing!'}), 401
        try:
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            current_user = User.query.filter_by(id=data['id']).first()
        except:
            return jsonify({'message': 'Token is invalid!'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

# Routes
@app.route('/register', methods=['POST'])
def register_user():
    data = request.json
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    new_user = User(username=data['username'], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully.'})

@app.route('/login', methods=['POST'])
def login_user():
    data = request.json
    user = User.query.filter_by(username=data['username']).first()
    if user and check_password_hash(user.password, data['password']):
        token = jwt.encode({'id': user.id,
                            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)},
                            app.config['SECRET_KEY'], algorithm="HS256")
        return jsonify({'token': token})
    return jsonify({'message': 'Invalid username or password.'}), 401

@app.route('/models', methods=['GET'])
def list_models():
    if models:
        available_models = list(models.keys())
        print("Available models:", available_models)
        return jsonify({'available_models': available_models}), 200
    else:
        return jsonify({'message': 'No models available'}), 204
@app.route('/predict', methods=['POST'])
@token_required
def predict_image(current_user):
    data = request.json
    if 'image' not in data or 'model_name' not in data:
        return jsonify({'message': 'Missing image or model name'}), 400
    image_data = data['image']
    model_name = data['model_name']
    if not image_data:
        return jsonify({'message': 'Empty image data'}), 400
    if model_name not in models:
        return jsonify({'message': 'Model not available'}), 404

    model = models[model_name]  # Load model if not already loaded
    if not model:
        return jsonify({'message': 'Error loading model'}), 500

    # Assuming decode_and_process_image handles model-specific preprocessing
    image = decode_and_preprocess_image(image_data, models[model_name]['config'])
    prediction = model.serve(image)
    predicted_class = np.argmax(prediction, axis=-1)[0]
    return jsonify({'prediction': str(predicted_class)})

@app.route('/history', methods=['GET'])
@token_required
def get_history(current_user):
    logs = RequestLog.query.filter_by(user_id=current_user.id).all()
    output = []
    for log in logs:
        log_data = {
            'image_name': log.image_name,
            'prediction_result': log.prediction_result,
            'timestamp': log.timestamp
        }
        output.append(log_data)
    return jsonify({'history': output})

# Main block
if __name__ == '__main__':
    # Ensure necessary directories exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Initialize DB
    with app.app_context():
        db.create_all()

    app.run(debug=True)
