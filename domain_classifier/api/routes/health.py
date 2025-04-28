import logging
from flask import jsonify

# Set up logging
logger = logging.getLogger(__name__)

def register_health_routes(app, llm_classifier, snowflake_conn, vector_db_conn):
    """Register health check related routes."""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Simple health check endpoint"""
        return jsonify({
            "status": "ok", 
            "llm_available": llm_classifier is not None,
            "snowflake_connected": getattr(snowflake_conn, 'connected', False),
            "vector_db_connected": getattr(vector_db_conn, 'connected', False)
        }), 200
        
    return app
