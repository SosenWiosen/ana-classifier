import os

from app import create_app

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))  # Default to port 5000 if PORT isn't set
    app.run(debug=True, port=port)
