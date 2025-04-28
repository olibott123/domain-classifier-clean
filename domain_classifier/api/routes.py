"""Routes module for the domain classifier API."""
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)

# Import all routes from the routes subpackage
from domain_classifier.api.routes import register_all_routes

# Initialize services
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
try:
    from domain_classifier.classifiers.llm_classifier import LLMClassifier
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
    from domain_classifier.storage.snowflake_connector import SnowflakeConnector
    snowflake_conn = SnowflakeConnector()
    if not getattr(snowflake_conn, 'connected', False):
        logger.warning("Snowflake connection failed, using fallback")
except Exception as e:
    logger.error(f"Error initializing Snowflake connector: {e}")
    from domain_classifier.storage.fallback_connector import FallbackSnowflakeConnector
    snowflake_conn = FallbackSnowflakeConnector()

# Initialize Vector DB connector
try:
    from domain_classifier.storage.vector_db import VectorDBConnector
    vector_db_conn = VectorDBConnector()
    logger.info(f"Vector DB connector initialized and connected: {getattr(vector_db_conn, 'connected', False)}")
except Exception as e:
    logger.error(f"Error initializing Vector DB connector: {e}")
    vector_db_conn = None

def register_routes(app):
    """Register all routes with the app (now delegates to the routes subpackage)."""
    
    return register_all_routes(app, llm_classifier, snowflake_conn, vector_db_conn)
