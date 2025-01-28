import os
from sqlite3 import OperationalError

import datetime
import jwt
from app import db
from app.models import User
from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash

bp = Blueprint('auth', __name__)


@bp.route('/register', methods=['POST'])
def register_user():
    data = request.json
    hashed_password = generate_password_hash(data['password'], method='pbkdf2:sha256')
    new_user = User(username=data['username'], password=hashed_password)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully.'})


@bp.route('/login', methods=['POST'])
def login_user():
    data = request.json
    print("login_user data: ", data)
    try:
        # Query the user from the database
        user = User.query.filter_by(username=data['username']).first()
        if not user or not check_password_hash(user.password, data['password']):
            return jsonify({"error": "Invalid username or password."}), 401

        # Create a token
        token = jwt.encode(
            {
                "id": user.id,
                "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
            },
            os.getenv("SECRET_KEY"),
            algorithm="HS256",
        )
        return jsonify({"token": token})

    except OperationalError as e:
        # Log the error for debugging
        print(e)
        return jsonify({"error": "An internal server error occurred. Please contact support." }), 500

    except Exception as e:
        print(e)
        # Catch any other unexpected errors
        return jsonify({"error": "An unexpected error occurred. Please try again later."}), 500
