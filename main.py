"""
Domain Classifier Main Entry Point

This is the main entry point for the Domain Classifier application.
It creates and configures the Flask application using the modular structure.
"""
import os
import logging
from domain_classifier.api.app import create_app

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
app = create_app()

if __name__ == "__main__":
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
