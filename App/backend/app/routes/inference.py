from flask import Blueprint, request, jsonify
from app.utils.token_decorator import token_required  # CORRECTfrom app.models import RequestLog
from app import db
from app.utils.load_models import load_models
from app.utils.preprocess import decode_and_preprocess_image
import json, os

bp = Blueprint('inference', __name__)
models = load_models(os.getenv("MODELS_FOLDER"))

@bp.route('/models', methods=['GET'])
@token_required
def list_models(current_user):
    if models:
        print(models)
        return jsonify({'available_models': list(models.keys())}), 200
    return jsonify({'message': 'No models available'}), 204

@bp.route('/predict', methods=['POST'])
@token_required
def predict_image(current_user):
    data = request.json
    model_name = data.get('model_name')
    image_data = data.get('image')

    if not model_name or not image_data:
        return jsonify({'message': 'Missing model name or image data'}), 400

    if model_name not in models:
        return jsonify({'message': 'Model is not available'}), 404

    model_info = models[model_name]
    try:
        image = decode_and_preprocess_image(image_data, model_info['config'])
        prediction = model_info['loaded_model'].serve(image)  # You may need to adjust this
        labels = model_info['labels']
        results = [{"label": label, "probability": float(prob)} for label, prob in zip(labels, prediction[0])]
        return jsonify({'prediction': results}), 200
    except Exception as e:
        return jsonify({'message': f'Error during prediction: {str(e)}'}), 500
