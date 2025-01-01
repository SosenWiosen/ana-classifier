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

@app.route('/predict', methods=['POST'])
@token_required
def predict_image(current_user):
    if 'file' not in request.files:
        return jsonify({'message': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load Model
    model = load_model('your_model.h5')  # Replace with the path to your Keras model

    # Preprocess Image for Prediction
    img = image.load_img(filepath, target_size=(224, 224))  # Change target_size as per your model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize if required by your model

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=-1)[0]

    # Log Request and Prediction Result
    new_request = RequestLog(user_id=current_user.id, image_name=filename, prediction_result=str(predicted_class))
    db.session.add(new_request)
    db.session.commit()

    return jsonify({'file': filename, 'prediction': str(predicted_class)})

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