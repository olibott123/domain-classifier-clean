from flask_cors import CORS

def setup_cors(app):
    """Set up CORS middleware for the application."""
    CORS(app)
    return app
