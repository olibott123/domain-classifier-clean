"""Flask application creation and configuration."""
import os
import logging
from flask import Flask
from domain_classifier.api.middleware import setup_cors
from domain_classifier.config.settings import get_port

# Import configuration and services
from domain_classifier.classifiers.llm_classifier import LLMClassifier
from domain_classifier.storage.snowflake_connector import SnowflakeConnector
from domain_classifier.storage.vector_db import VectorDBConnector

# Set up logging
logger = logging.getLogger(__name__)

def create_app():
    """
    Create and configure the Flask application.
    
    Returns:
        Flask: Configured Flask application
    """
    # Initialize the Flask application
    app = Flask(__name__)
    
    # Set up middleware
    setup_cors(app)
    
    # Initialize services
    # Get API key from environment
    ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
    
    try:
        # Initialize LLM Classifier
        llm_classifier = LLMClassifier(
            api_key=ANTHROPIC_API_KEY,
            model="claude-3-haiku-20240307"
        )
        logger.info(f"Initialized LLM classifier with model: claude-3-haiku-20240307")
    except Exception as e:
        logger.error(f"Failed to initialize LLM classifier: {e}")
        llm_classifier = None

    # Initialize Snowflake connector
    try:
        snowflake_conn = SnowflakeConnector()
        if not getattr(snowflake_conn, 'connected', False):
            logger.warning("Snowflake connection failed, using fallback")
    except Exception as e:
        logger.error(f"Error initializing Snowflake connector: {e}")
        from domain_classifier.storage.fallback_connector import FallbackSnowflakeConnector
        snowflake_conn = FallbackSnowflakeConnector()

    # Initialize Vector DB connector
    try:
        vector_db_conn = VectorDBConnector()
        logger.info(f"Vector DB connector initialized and connected: {getattr(vector_db_conn, 'connected', False)}")
    except Exception as e:
        logger.error(f"Error initializing Vector DB connector: {e}")
        vector_db_conn = None

    # Import routes after initializing services to avoid circular imports
    from domain_classifier.api.routes import register_routes

    # Register routes with all the services
    app = register_routes(app, llm_classifier, snowflake_conn, vector_db_conn)
    
    # Set up JSON encoder
    from domain_classifier.utils.json_encoder import CustomJSONEncoder
    app.json_encoder = CustomJSONEncoder
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = get_port()
    app.run(host='0.0.0.0', port=port)
