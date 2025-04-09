import snowflake.connector
import traceback
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeConnector:
    def __init__(self):
        self.conn_params = {
            'user': 'url_domain_crawler_testing_user',
            'private_key_path': '/workspace/rsa_key.der',  # Updated path
            'account': 'DOMOTZ-MAIN',
            'warehouse': 'TESTING_WH',
            'database': 'DOMOTZ_TESTING_SOURCE',
            'schema': 'EXTERNAL_PUSH',
            'authenticator': 'snowflake_jwt',
            'session_parameters': {'QUERY_TAG': 'WebCrawlerBot'}
        }

    def load_private_key(self, path):
        try:
            with open(path, "rb") as key_file:
                return key_file.read()
        except Exception as e:
            logger.error(f"Error loading private key from {path}: {e}")
            # Try fallback path
            fallback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rsa_key.der')
            logger.info(f"Trying fallback path: {fallback_path}")
            try:
                with open(fallback_path, "rb") as key_file:
                    return key_file.read()
            except Exception as fallback_e:
                logger.error(f"Fallback path also failed: {fallback_e}")
                raise

    def get_connection(self):
        try:
            private_key = self.load_private_key(self.conn_params['private_key_path'])
            return snowflake.connector.connect(
                user=self.conn_params['user'],
                account=self.conn_params['account'],
                private_key=private_key,
                warehouse=self.conn_params['warehouse'],
                database=self.conn_params['database'],
                schema=self.conn_params['schema'],
                authenticator=self.conn_params['authenticator'],
                session_parameters=self.conn_params['session_parameters']
            )
        except Exception as e:
            logger.error(f"Error connecting to Snowflake: {e}")
            # For now, return None instead of raising to allow API to function without Snowflake
            return None

    def save_domain_content(self, domain, url, content):
        try:
            conn = self.get_connection()
            if not conn:
                logger.warning("Snowflake connection failed - skipping domain content save")
                return False, "Connection failed"
                
            cursor = conn.cursor()
            
            # Check if content already exists
            cursor.execute("""
                SELECT COUNT(*) FROM DOMAIN_CONTENT WHERE domain = %s
            """, (domain,))
            
            if cursor.fetchone()[0] == 0:
                logger.info(f"Inserting new domain content for {domain}")
                cursor.execute("""
                    INSERT INTO DOMAIN_CONTENT (domain, url, text_content, crawl_date)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP())
                """, (domain, url, content))
                conn.commit()
                logger.info(f"Successfully saved domain content for {domain}")
            else:
                logger.info(f"Content for domain {domain} already exists, skipping insertion")
                
            cursor.close()
            conn.close()
            return True, None
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"Error saving domain content: {error_msg}")
            return False, error_msg

    def save_classification(self, domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method):
        try:
            logger.info(f"Attempting to save classification for {domain}: {company_type} ({confidence_score})")
            conn = self.get_connection()
            if not conn:
                logger.warning("Snowflake connection failed - skipping classification save")
                return False, "Connection failed"
                
            cursor = conn.cursor()
            
            insert_query = """
                INSERT INTO DOMAIN_CLASSIFICATION (
                    DOMAIN, 
                    COMPANY_TYPE, 
                    CONFIDENCE_SCORE, 
                    ALL_SCORES, 
                    LOW_CONFIDENCE, 
                    DETECTION_METHOD, 
                    MODEL_METADATA, 
                    CLASSIFICATION_DATE
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
            """
            
            # Print the query parameters for debugging
            params = (domain, company_type, confidence_score, all_scores, low_confidence, detection_method, model_metadata)
            logger.info(f"Insert parameters: {params}")
            
            cursor.execute(insert_query, params)
            conn.commit()
            
            # Verify the insert was successful
            cursor.execute("SELECT COUNT(*) FROM DOMAIN_CLASSIFICATION WHERE DOMAIN = %s", (domain,))
            count = cursor.fetchone()[0]
            logger.info(f"After insert, found {count} records for domain {domain}")
            
            cursor.close()
            conn.close()
            logger.info(f"Successfully saved classification for {domain}")
            return True, None
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"Error saving classification: {error_msg}")
            return False, error_msg
            
    def check_existing_classification(self, domain):
        try:
            logger.info(f"Checking for existing classification for domain: {domain}")
            conn = self.get_connection()
            if not conn:
                logger.warning("Snowflake connection failed - skipping existence check")
                return None
                
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
                logger.info(f"Found existing classification for {domain}: {result[1]}")
                existing_record = {
                    "domain": result[0],
                    "company_type": result[1],
                    "confidence_score": result[2],
                    "all_scores": result[3] if len(result) > 3 else "{}",
                    "low_confidence": result[4] if len(result) > 4 else True,
                    "detection_method": result[5] if len(result) > 5 else "unknown",
                    "model_metadata": result[6] if len(result) > 6 else "{}",
                    "classification_date": str(result[7]) if len(result) > 7 else ""
                }
                cursor.close()
                conn.close()
                return existing_record
            else:
                logger.info(f"No existing classification found for {domain}")
                cursor.close()
                conn.close()
                return None
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"Error checking existing classification: {error_msg}")
            return None
