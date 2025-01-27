from flask import Blueprint

auth_bp = Blueprint('auth', __name__)
inference_bp = Blueprint('inference', __name__)
history_bp = Blueprint('history', __name__)

from .auth import bp as auth
from .inference import bp as inference
from .history import bp as history
