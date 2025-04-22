"""
Domain Classifier Main Entry Point
This is the main entry point for the Domain Classifier application.
It creates and configures the Flask application using the modular structure.
"""
import os
import sys
import logging
import traceback

# Add very visible debug statements to verify this file is being used
print("="*80)
print("STARTING DOMAIN CLASSIFIER USING NEW MODULAR STRUCTURE")
print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir("."))
print("="*80)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("INITIALIZING DOMAIN CLASSIFIER USING NEW MODULAR STRUCTURE")

try:
    # Import from the modular structure
    from domain_classifier.api.app import create_app
    logger.info("Successfully imported create_app from modular structure")
except Exception as e:
    print("ERROR IMPORTING FROM MODULAR STRUCTURE:", str(e))
    logger.error(f"Error importing from modular structure: {e}")
    logger.error(traceback.format_exc())
    # Fall back to old structure for now
    print("Falling back to old structure")
    import api_service
    app = api_service.app
    logger.info("Fallback to old structure completed")
else:
    # Create RSA key from base64 environment variable (previously in startup.sh)
    if "SNOWFLAKE_KEY_BASE64" in os.environ:
        import base64
        key_path = "/workspace/rsa_key.der"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(key_path), exist_ok=True)
        
        # Write key file
        try:
            with open(key_path, "wb") as key_file:
                key_file.write(base64.b64decode(os.environ["SNOWFLAKE_KEY_BASE64"]))
            os.chmod(key_path, 0o600)
            logger.info(f"Created Snowflake key file at {key_path}")
        except Exception as e:
            logger.error(f"Failed to create Snowflake key file: {e}")
    else:
        logger.warning("SNOWFLAKE_KEY_BASE64 not set. Snowflake integration will be disabled.")
    
    # Create the Flask application
    logger.info("Creating Flask app using modular structure")
    app = create_app()
    logger.info("Flask app created successfully")

# Print debug info about the app
print("App routes:")
for rule in app.url_map.iter_rules():
    print(f"  - {rule}")

# For direct execution
if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting app on port {port}")
    app.run(host="0.0.0.0", port=port)
else:
    logger.info("Running as imported module (e.g., via gunicorn)")
