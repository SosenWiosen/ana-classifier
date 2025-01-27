import os

from app import create_app, db

app = create_app()

if __name__ == '__main__':
    if os.getenv('FLASK_ENV') == 'development':
        with app.app_context():
            db.create_all()  # Automatically create missing tables in development
            print("All tables created!")
    port = int(os.getenv("PORT", 5000))  # Default to port 5000 if PORT isn't set
    app.run(debug=True, port=port)
