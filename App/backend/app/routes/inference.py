import json
import os

from app import db  # Import the SQLAlchemy instance
from app.models import RequestLog
from app.utils.load_models import load_models
from app.utils.preprocess import decode_and_preprocess_image
from app.utils.token_decorator import token_required
from flask import Blueprint, request, jsonify

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
    image_filename = data.get('image_filename', "unknown")  # Optional: Add image filename

    if not model_name or not image_data:
        return jsonify({'message': 'Missing model name or image data'}), 400
    if model_name not in models:
        return jsonify({'message': 'Model is not available'}), 404

    model_info = models[model_name]
    try:
        # Decode and preprocess the image
        image = decode_and_preprocess_image(image_data, model_info['config'])

        # Perform inference on the image using the selected model
        prediction = model_info['loaded_model'].serve(image)
        labels = model_info['labels']

        # Extract results and format them
        results = [
            {"label": label, "probability": float(prob)}
            for label, prob in zip(labels, prediction[0])
        ]

        # Save the prediction result into the database
        prediction_result = json.dumps({"prediction": results})  # Convert the results to JSON string
        new_log = RequestLog(
            user_id=current_user.id,  # Get user ID from the current_user object
            image_name=image_filename,  # Image name (if provided in the request)
            prediction_result=prediction_result,
        )
        db.session.add(new_log)
        db.session.commit()

        # Return the prediction results
        return jsonify({'prediction': results}), 200

    except Exception as e:
        # Handle errors gracefully
        return jsonify({'message': f'Error during prediction: {str(e)}'}), 500
