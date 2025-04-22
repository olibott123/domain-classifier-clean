from flask import Flask
from domain_classifier.api.middleware import setup_cors
from domain_classifier.api.routes import register_routes
from domain_classifier.config.settings import get_port

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)
    
    # Set up middleware
    setup_cors(app)
    
    # Register routes
    register_routes(app)
    
    # Set up JSON encoder
    from domain_classifier.utils.json_encoder import CustomJSONEncoder
    app.json_encoder = CustomJSONEncoder
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = get_port()
    app.run(host='0.0.0.0', port=port)
