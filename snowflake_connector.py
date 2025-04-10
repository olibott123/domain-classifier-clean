import os
import json
import logging
import time
from datetime import datetime
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
        self.database = os.environ.get('SNOWFLAKE_DATABASE', 'DOMAIN_CLASSIFIER')
        self.schema = os.environ.get('SNOWFLAKE_SCHEMA', 'PUBLIC')
        self.warehouse = os.environ.get('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
        self.role = os.environ.get('SNOWFLAKE_ROLE', 'ACCOUNTADMIN')
        
        # Check for required credentials
        if not self.account or not self.user:
            logger.warning("Missing Snowflake credentials. Using fallback mode.")
            return
        
        # Try to load private key for authentication
        self.private_key_path = os.environ.get('SNOWFLAKE_PRIVATE_KEY_PATH', '/workspace/rsa_key.der')
        self.private_key_passphrase = os.environ.get('SNOWFLAKE_PRIVATE_KEY_PASSPHRASE')
        
        try:
            self._init_connection()
        except Exception as e:
            logger.error(f"Failed to initialize Snowflake connection: {e}")
            self.connected = False
    
    def _init_connection(self):
        """Initialize the Snowflake connection."""
        try:
            # Prepare private key if it exists
            private_key = None
            if os.path.exists(self.private_key_path):
                with open(self.private_key_path, 'rb') as key_file:
                    private_key_data = key_file.read()
                    private_key = load_der_private_key(
                        private_key_data,
                        password=self.private_key_passphrase.encode() if self.private_key_passphrase else None,
                        backend=default_backend()
                    )
            
            # Connect to Snowflake
            conn_params = {
                'user': self.user,
                'account': self.account,
                'database': self.database,
                'schema': self.schema,
                'warehouse': self.warehouse,
                'role': self.role
            }
            
            if private_key:
                conn_params['private_key'] = private_key
            
            # Test connection
            conn = snowflake.connector.connect(**conn_params)
            cursor = conn.cursor()
            cursor.execute("SELECT current_version()")
            version = cursor.fetchone()[0]
            logger.info(f"Connected to Snowflake. Version: {version}")
            
            # Create necessary tables if they don't exist
            self._create_tables(cursor)
            
            cursor.close()
            conn.close()
            self.connected = True
        except Exception as e:
            logger.error(f"Error connecting to Snowflake: {e}")
            self.connected = False
            raise
    
    def _create_tables(self, cursor):
        """Create necessary tables if they don't exist."""
        try:
            # Domain content table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS DOMAIN_CONTENT (
                domain VARCHAR(255) PRIMARY KEY,
                url VARCHAR(1000),
                content VARCHAR(16777216),
                last_crawled TIMESTAMP_LTZ
            )
            """)
            
            # Domain classification table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS DOMAIN_CLASSIFICATION (
                domain VARCHAR(255) PRIMARY KEY,
                company_type VARCHAR(255),
                confidence_score FLOAT,
                all_scores VARCHAR(16777216),
                model_metadata VARCHAR(16777216),
                low_confidence BOOLEAN,
                detection_method VARCHAR(255),
                classified_at TIMESTAMP_LTZ
            )
            """)
            
            logger.info("Created necessary tables in Snowflake")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def _get_connection(self):
        """Get a new Snowflake connection."""
        if not self.connected:
            logger.warning("Not connected to Snowflake")
            return None
            
        try:
            conn_params = {
                'user': self.user,
                'account': self.account,
                'database': self.database,
                'schema': self.schema,
                'warehouse': self.warehouse,
                'role': self.role
            }
            
            # Load private key if it exists
            if os.path.exists(self.private_key_path):
                with open(self.private_key_path, 'rb') as key_file:
                    private_key_data = key_file.read()
                    private_key = load_der_private_key(
                        private_key_data,
                        password=self.private_key_passphrase.encode() if self.private_key_passphrase else None,
                        backend=default_backend()
                    )
                conn_params['private_key'] = private_key
            
            return snowflake.connector.connect(**conn_params)
        except Exception as e:
            logger.error(f"Error getting Snowflake connection: {e}")
            return None
    
    def check_existing_classification(self, domain):
        """Check if a domain already has a classification in Snowflake."""
        if not self.connected:
            logger.info(f"Fallback: No existing classification for {domain}")
            return None
            
        conn = self._get_connection()
        if not conn:
            return None
            
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    domain, 
                    company_type, 
                    confidence_score, 
                    all_scores, 
                    model_metadata, 
                    low_confidence,
                    detection_method
                FROM DOMAIN_CLASSIFICATION 
                WHERE domain = %s
            """, (domain,))
            
            row = cursor.fetchone()
            if row:
                column_names = [desc[0].lower() for desc in cursor.description]
                result = dict(zip(column_names, row))
                logger.info(f"Found existing classification for {domain}: {result['company_type']}")
                return result
            
            logger.info(f"No existing classification found for {domain}")
            return None
        except Exception as e:
            logger.error(f"Error checking for existing classification: {e}")
            return None
        finally:
            if conn:
                conn.close()
    
    def save_domain_content(self, domain, url, content):
        """Save domain content to Snowflake."""
        if not self.connected:
            logger.info(f"Fallback: Not saving domain content for {domain}")
            return True, None
            
        conn = self._get_connection()
        if not conn:
            return False, "Failed to connect to Snowflake"
            
        try:
            cursor = conn.cursor()
            
            # Check if domain already exists
            cursor.execute("""
                SELECT COUNT(*) FROM DOMAIN_CONTENT WHERE domain = %s
            """, (domain,))
            
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Update existing record
                cursor.execute("""
                    UPDATE DOMAIN_CONTENT
                    SET url = %s, content = %s, last_crawled = CURRENT_TIMESTAMP()
                    WHERE domain = %s
                """, (url, content, domain))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO DOMAIN_CONTENT (domain, url, content, last_crawled)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP())
                """, (domain, url, content))
            
            conn.commit()
            logger.info(f"Saved domain content for {domain}")
            return True, None
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error saving domain content: {e}")
            return False, str(e)
        finally:
            if conn:
                conn.close()
    
    def save_classification(self, domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method):
        """Save domain classification to Snowflake."""
        if not self.connected:
            logger.info(f"Fallback: Not saving classification for {domain}")
            return True, None
            
        conn = self._get_connection()
        if not conn:
            return False, "Failed to connect to Snowflake"
            
        try:
            cursor = conn.cursor()
            
            # Check if domain already exists
            cursor.execute("""
                SELECT COUNT(*) FROM DOMAIN_CLASSIFICATION WHERE domain = %s
            """, (domain,))
            
            count = cursor.fetchone()[0]
            
            if count > 0:
                # Update existing record
                cursor.execute("""
                    UPDATE DOMAIN_CLASSIFICATION
                    SET company_type = %s, 
                        confidence_score = %s, 
                        all_scores = %s, 
                        model_metadata = %s,
                        low_confidence = %s,
                        detection_method = %s,
                        classified_at = CURRENT_TIMESTAMP()
                    WHERE domain = %s
                """, (company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method, domain))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO DOMAIN_CLASSIFICATION 
                    (domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method, classified_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
                """, (domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method))
            
            conn.commit()
            logger.info(f"Saved classification for {domain}: {company_type}")
            return True, None
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Error saving classification: {e}")
            return False, str(e)
        finally:
            if conn:
                conn.close()
