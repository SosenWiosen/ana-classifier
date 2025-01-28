import os

from app import create_app

app = create_app()

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5002))  # Default to port 5000 if PORT isn't set
    app.run(debug=True, host='0.0.0.0', port=port)
