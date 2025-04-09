import snowflake.connector
import traceback
import logging
import os
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SnowflakeConnector:
    def __init__(self):
        # Use environment variables or fallback to hardcoded parameters
        self.conn_params = {
            'user': os.environ.get('SNOWFLAKE_USER', 'url_domain_crawler_testing_user'),
            'private_key_path': os.environ.get('SNOWFLAKE_PRIVATE_KEY_PATH', '/root/crawler_api/rsa_key.der'),
            'account': os.environ.get('SNOWFLAKE_ACCOUNT', 'DOMOTZ-MAIN'),
            'warehouse': os.environ.get('SNOWFLAKE_WAREHOUSE', 'TESTING_WH'),
            'database': os.environ.get('SNOWFLAKE_DATABASE', 'DOMOTZ_TESTING_SOURCE'),
            'schema': os.environ.get('SNOWFLAKE_SCHEMA', 'EXTERNAL_PUSH'),
            'authenticator': 'snowflake_jwt',
            'session_parameters': {'QUERY_TAG': 'WebCrawlerBot'}
        }

    def load_private_key(self, path):
        with open(path, "rb") as key_file:
            return key_file.read()

    def get_connection(self):
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

    def save_domain_content(self, domain, url, content):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Truncate content if it's too long
            max_content_length = 100000  # Adjust as needed
            content = content[:max_content_length]

            # Check if content already exists
            cursor.execute("""
                MERGE INTO DOMAIN_CONTENT AS target
                USING (SELECT %s AS domain, %s AS url, %s AS text_content, CURRENT_TIMESTAMP() AS crawl_date) AS source
                ON target.domain = source.domain
                WHEN MATCHED THEN
                    UPDATE SET 
                        url = source.url,
                        text_content = source.text_content,
                        crawl_date = source.crawl_date
                WHEN NOT MATCHED THEN
                    INSERT (domain, url, text_content, crawl_date)
                    VALUES (source.domain, source.url, source.text_content, source.crawl_date)
            """, (domain, url, content))
            
            conn.commit()
            cursor.close()
            conn.close()
            logger.info(f"Successfully saved/updated domain content for {domain}")
            return True, None
        except Exception as e:
            error_msg = traceback.format_exc()
            logger.error(f"Error saving domain content: {error_msg}")
            return False, error_msg

    def save_classification(self, domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Ensure all_scores and model_metadata are JSON strings
            all_scores_json = json.dumps(all_scores) if isinstance(all_scores, dict) else all_scores
            model_metadata_json = json.dumps(model_metadata) if isinstance(model_metadata, dict) else model_metadata

            insert_query = """
                MERGE INTO DOMAIN_CLASSIFICATION AS target
                USING (
                    SELECT 
                        %s AS domain, 
                        %s AS company_type, 
                        %s AS confidence_score, 
                        %s AS all_scores, 
                        %s AS model_metadata, 
                        %s AS low_confidence, 
                        %s AS detection_method, 
                        CURRENT_TIMESTAMP() AS classification_date
                ) AS source
                ON target.domain = source.domain
                WHEN MATCHED THEN
                    UPDATE SET 
                        company_type = source.company_type,
                        confidence_score = source.confidence_score,
                        all_scores = source.all_scores,
                        model_metadata = source.model_metadata,
                        low_confidence = source.low_confidence,
                        detection_method = source.detection_method,
                        classification_date = source.classification_date
                WHEN NOT MATCHED THEN
                    INSERT (
                        domain, company_type, confidence_score, 
                        all_scores, model_metadata, low_confidence, 
                        detection_method, classification_date
                    )
                    VALUES (
                        source.domain, source.company_type, source.confidence_score, 
                        source.all_scores, source.model_metadata, source.low_confidence, 
                        source.detection_method, source.classification_date
                    )
            """
            
            params = (
                domain, company_type, confidence_score, 
                all_scores_json, model_metadata_json, 
                low_confidence, detection_method
            )
            
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
            cursor = conn.cursor()
            query = """
                SELECT
                    DOMAIN,
                    COMPANY_TYPE,
                    CONFIDENCE_SCORE,
                    ALL_SCORES,
                    LOW_CONFIDENCE,
                    DETECTION_METHOD,
                    CLASSIFICATION_DATE
                FROM DOMAIN_CLASSIFICATION
                WHERE DOMAIN = %s
                ORDER BY CLASSIFICATION_DATE DESC
                LIMIT 1
            """
            cursor.execute(query, (domain,))
            result = cursor.fetchone()
            
            if result:
                existing_record = {
                    "domain": result[0],
                    "company_type": result[1],
                    "confidence_score": result[2],
                    "all_scores": json.loads(result[3]) if result[3] else {},
                    "low_confidence": result[4],
                    "detection_method": result[5],
                    "classification_date": str(result[6])
                }
                logger.info(f"Found existing classification for {domain}: {existing_record['company_type']}")
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
