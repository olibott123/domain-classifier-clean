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
        """Initialize Snowflake connection with environment variables or defaults."""
        self.connected = False
        
        # Get connection parameters from environment variables or use defaults from your old code
        self.account = os.environ.get('SNOWFLAKE_ACCOUNT', 'DOMOTZ-MAIN')
        self.user = os.environ.get('SNOWFLAKE_USER', 'url_domain_crawler_testing_user')
        self.database = os.environ.get('SNOWFLAKE_DATABASE', 'DOMOTZ_TESTING_SOURCE')
        self.schema = os.environ.get('SNOWFLAKE_SCHEMA', 'EXTERNAL_PUSH')
        self.warehouse = os.environ.get('SNOWFLAKE_WAREHOUSE', 'TESTING_WH')
        self.role = os.environ.get('SNOWFLAKE_ROLE', 'ACCOUNTADMIN')
        self.authenticator = os.environ.get('SNOWFLAKE_AUTHENTICATOR', 'snowflake_jwt')
        
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
                    
                    # Create necessary tables if they don't exist
                    self._create_tables(cursor)
                    
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
    
    def _create_tables(self, cursor):
        """Create necessary tables if they don't exist."""
        try:
            # Domain content table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS DOMAIN_CONTENT (
                domain VARCHAR(255) PRIMARY KEY,
                url VARCHAR(1000),
                text_content VARCHAR(16777216),
                crawl_date TIMESTAMP_LTZ
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
                classification_date TIMESTAMP_LTZ
            )
            """)
            
            logger.info("Created necessary tables in Snowflake")
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
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
            cursor.execute("""
                SELECT
                    DOMAIN,
                    COMPANY_TYPE,
                    CONFIDENCE_SCORE,
                    ALL_SCORES,
                    MODEL_METADATA,
                    LOW_CONFIDENCE,
                    DETECTION_METHOD,
                    CLASSIFICATION_DATE
                FROM DOMAIN_CLASSIFICATION
                WHERE DOMAIN = %s
                ORDER BY CLASSIFICATION_DATE DESC
                LIMIT 1
            """, (domain,))
            
            result = cursor.fetchone()
            if result:
                column_names = ["domain", "company_type", "confidence_score", "all_scores", 
                                "model_metadata", "low_confidence", "detection_method", "classification_date"]
                existing_record = dict(zip(column_names, result))
                logger.info(f"Found existing classification for {domain}: {existing_record['company_type']}")
                return existing_record
            
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
            
        conn = self.get_connection()
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
                    SET url = %s, text_content = %s, crawl_date = CURRENT_TIMESTAMP()
                    WHERE domain = %s
                """, (url, content, domain))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO DOMAIN_CONTENT (domain, url, text_content, crawl_date)
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
            
        conn = self.get_connection()
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
                        classification_date = CURRENT_TIMESTAMP()
                    WHERE domain = %s
                """, (company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method, domain))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO DOMAIN_CLASSIFICATION 
                    (domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method, classification_date)
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
