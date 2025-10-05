from waitress import serve
from app import app  # make sure your Flask app object is named 'app'

if __name__ == '__main__':
    print("🚀 Starting Waitress server...")
    serve(app, host='0.0.0.0', port=5000)
