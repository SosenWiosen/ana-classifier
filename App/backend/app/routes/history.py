from flask import Blueprint, jsonify
from app.models import RequestLog
from app.utils.token_decorator import token_required

bp = Blueprint('history', __name__)

@bp.route('/history', methods=['GET'])
@token_required
def get_history(current_user):
    logs = RequestLog.query.filter_by(user_id=current_user.id).all()
    if not logs:
        return jsonify({'message': 'No history found.'}), 204

    output = []
    for log in logs:
        output.append({
            'image_name': log.image_name,
            'prediction_result': log.prediction_result,
            'timestamp': log.timestamp
        })
    return jsonify({'history': output}), 200
