from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import uuid
import jwt
import datetime
from functools import wraps
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

import base64  # For handling base64 encoding
from PIL import Image  # For image manipulation
from io import BytesIO  # For handling image as a byte buffer

app = Flask(__name__)
CORS(app)
# Configuration
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = './uploads'

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

app.config['UPLOAD_FOLDER'] = "path_to_upload_folder"
app.config['MODELS_FOLDER'] = "backend/models/"

models = {}  # This will store model instances

def load_models():
    models_folder = app.config['MODELS_FOLDER']
    for model_dir in os.listdir(models_folder):
        config_path = os.path.join(models_folder, model_dir, 'model_config.json')
        if os.path.isfile(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Construct the full path to the model file
                model_file_path = os.path.join(models_folder, model_dir, config['model_path'])
                model_name = config['name']  # Use the name from config
                if os.path.isfile(model_file_path):
                    models[model_name] = {
                        'model': load_model(model_file_path),
                        'config': config
                    }
                    print(f"Loaded model {model_name} from {model_file_path}")

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

    # Assuming decode_and_process_image handles model-specific preprocessing
    image = decode_and_process_image(image_data, models[model_name]['config'])

    model = models[model_name]['model']
    prediction = model.predict(image)
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
