import os
import json
import logging
import traceback
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.serialization import load_der_private_key
import snowflake.connector
from snowflake.connector.errors import ProgrammingError, DatabaseError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeConnector:
    def __init__(self):
        """Initialize Snowflake connection with environment variables."""
        self.connected = False
        
        # Get connection parameters from environment variables
        self.account = os.environ.get('SNOWFLAKE_ACCOUNT')
        self.user = os.environ.get('SNOWFLAKE_USER')
        self.database = os.environ.get('SNOWFLAKE_DATABASE')
        self.schema = os.environ.get('SNOWFLAKE_SCHEMA')
        self.warehouse = os.environ.get('SNOWFLAKE_WAREHOUSE')
        self.authenticator = os.environ.get('SNOWFLAKE_AUTHENTICATOR')
        self.private_key_path = os.environ.get('SNOWFLAKE_PRIVATE_KEY_PATH')
        
        # Create RSA key file if it doesn't exist
        if not os.path.exists(self.private_key_path) and 'SNOWFLAKE_KEY_BASE64' in os.environ:
            try:
                dir_path = os.path.dirname(self.private_key_path)
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)
                    
                with open(self.private_key_path, 'wb') as key_file:
                    import base64
                    key_data = base64.b64decode(os.environ.get('SNOWFLAKE_KEY_BASE64'))
                    key_file.write(key_data)
                    
                os.chmod(self.private_key_path, 0o600)
                logger.info(f"Created RSA key file at {self.private_key_path}")
            except Exception as e:
                logger.error(f"Failed to create RSA key file: {e}")
        
        # Check for required credentials
        if not self.account or not self.user or not self.private_key_path:
            logger.warning("Missing Snowflake credentials. Using fallback mode.")
            return
        
        try:
            self._init_connection()
        except Exception as e:
            logger.error(f"Failed to initialize Snowflake connection: {e}")
            self.connected = False
    
    def _init_connection(self):
        """Initialize the Snowflake connection."""
        try:
            # Check if the RSA key exists
            if os.path.exists(self.private_key_path):
                logger.info(f"Found RSA key at {self.private_key_path}")
                
                # Test connection
                conn = self.get_connection()
                if conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT current_version()")
                    version = cursor.fetchone()[0]
                    logger.info(f"Connected to Snowflake. Version: {version}")
                    
                    # Skip table creation as tables already exist
                    cursor.close()
                    conn.close()
                    self.connected = True
                else:
                    logger.error("Could not establish Snowflake connection")
                    self.connected = False
            else:
                logger.warning(f"RSA key not found at {self.private_key_path}")
                self.connected = False
        except Exception as e:
            logger.error(f"Error connecting to Snowflake: {e}")
            self.connected = False
            raise
    
    def load_private_key(self, path):
        """Load private key from path."""
        try:
            with open(path, "rb") as key_file:
                return key_file.read()
        except Exception as e:
            logger.error(f"Error loading private key: {e}")
            return None
    
    def get_connection(self):
        """Get a new Snowflake connection."""
        if not os.path.exists(self.private_key_path):
            logger.warning(f"Private key not found at {self.private_key_path}")
            return None
            
        try:
            private_key = self.load_private_key(self.private_key_path)
            if not private_key:
                return None
                
            return snowflake.connector.connect(
                user=self.user,
                account=self.account,
                private_key=private_key,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
                authenticator=self.authenticator,
                session_parameters={'QUERY_TAG': 'WebCrawlerBot'}
            )
        except Exception as e:
            logger.error(f"Error getting Snowflake connection: {e}")
            return None
    
    def check_existing_classification(self, domain):
        """Check if a domain already has a classification in Snowflake."""
        if not self.connected:
            logger.info(f"Fallback: No existing classification for {domain}")
            return None
            
        conn = self.get_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            query = """
                SELECT
                    DOMAIN,
                    COMPANY_TYPE,
                    CONFIDENCE_SCORE,
                    ALL_SCORES,
                    LOW_CONFIDENCE,
                    DETECTION_METHOD,
                    MODEL_METADATA,
                    CLASSIFICATION_DATE
                FROM DOMAIN_CLASSIFICATION
                WHERE DOMAIN = %s
                ORDER BY CLASSIFICATION_DATE DESC
                LIMIT 1
            """
            cursor.execute(query, (domain,))
            
            result = cursor.fetchone()
            if result:
                column_names = ["domain", "company_type", "confidence_score", "all_scores", 
                                "low_confidence", "detection_method", "model_metadata", "classification_date"]
                existing_record = dict(zip(column_names, result))
                logger.info(f"Found existing classification for {domain}: {existing_record['company_type']}")
                return existing_record
            
            logger.info(f"No existing classification found for {domain}")
            return None
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"Error checking existing classification: {error_msg}")
            return None
        finally:
            if conn:
                conn.close()
    
    def save_domain_content(self, domain, url, content):
        """Save domain content to Snowflake."""
        if not self.connected:
            logger.info(f"Fallback: Not saving domain content for {domain}")
            return True, None
            
        conn = self.get_connection()
        if not conn:
            return False, "Failed to connect to Snowflake"
            
        try:
            cursor = conn.cursor()
            
            # Insert new record - using the table structure with ID autoincrement
            cursor.execute("""
                INSERT INTO DOMAIN_CONTENT (domain, url, text_content, crawl_date)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP())
            """, (domain, url, content))
            
            conn.commit()
            logger.info(f"Saved domain content for {domain}")
            return True, None
        except Exception as e:
            error_msg = traceback.format_exc()
            if conn:
                conn.rollback()
            logger.error(f"Error saving domain content: {error_msg}")
            return False, error_msg
        finally:
            if conn:
                conn.close()
    
    def save_classification(self, domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method):
        """Save domain classification to Snowflake."""
        if not self.connected:
            logger.info(f"Fallback: Not saving classification for {domain}")
            return True, None
            
        conn = self.get_connection()
        if not conn:
            return False, "Failed to connect to Snowflake"
            
        try:
            cursor = conn.cursor()
            
            # Insert new record - using the table structure with ID autoincrement
            cursor.execute("""
                INSERT INTO DOMAIN_CLASSIFICATION 
                (domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method, classification_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
            """, (domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method))
            
            conn.commit()
            logger.info(f"Saved classification for {domain}: {company_type}")
            return True, None
        except Exception as e:
            error_msg = traceback.format_exc()
            if conn:
                conn.rollback()
            logger.error(f"Error saving classification: {error_msg}")
            return False, error_msg
        finally:
            if conn:
                conn.close()
