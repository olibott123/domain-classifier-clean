import os
from flask import Flask
from domain_classifier.api.middleware import setup_cors
from domain_classifier.api.routes import register_all_routes
from domain_classifier.config.settings import get_port
from domain_classifier.classifiers.llm_classifier import LLMClassifier
from domain_classifier.storage.snowflake_connector import SnowflakeConnector
from domain_classifier.storage.vector_db import VectorDBConnector

def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Set up middleware
    setup_cors(app)

    # Initialize services
    llm_classifier = LLMClassifier(
        api_key=os.environ.get("ANTHROPIC_API_KEY"), 
        model="claude-3-haiku-20240307"
    )
    snowflake_conn = SnowflakeConnector()
    vector_db_conn = VectorDBConnector()

    # Register routes with initialized services
    register_all_routes(app, llm_classifier, snowflake_conn, vector_db_conn)

    # Set up JSON encoder
    from domain_classifier.utils.json_encoder import CustomJSONEncoder
    app.json_encoder = CustomJSONEncoder

    return app

if __name__ == '__main__':
    app = create_app()
    port = get_port()
    app.run(host='0.0.0.0', port=port)
