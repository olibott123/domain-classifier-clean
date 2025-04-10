from flask import Flask, request, jsonify
from flask_cors import CORS
from domain_classifier_fixed import DomainClassifier
from snowflake_connector import SnowflakeConnector
import requests
import time
from urllib.parse import urlparse
import json
import os
import numpy as np
import logging
import threading
import uuid
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Custom JSON encoder to handle problematic types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Configure Flask to use the custom encoder
app.json_encoder = CustomJSONEncoder

# Get API keys and settings from environment variables (no defaults)
APIFY_TASK_ID = os.environ.get("APIFY_TASK_ID")
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "domain-embeddings")

# Initialize Classifier with both Pinecone and LLM support
classifier = DomainClassifier(
    model_path="./domain_classifier_model_enhanced.pkl",
    use_pinecone=True,
    pinecone_api_key=PINECONE_API_KEY,
    pinecone_index_name=PINECONE_INDEX_NAME,
    use_llm=True,
    anthropic_api_key=ANTHROPIC_API_KEY,
    llm_model="claude-3-haiku-20240307"
)

# Initialize Snowflake connector
try:
    snowflake_conn = SnowflakeConnector()
    if not getattr(snowflake_conn, 'connected', False):
        logger.warning("Snowflake connection failed, using fallback")
except Exception as e:
    logger.error(f"Error initializing Snowflake connector: {e}")
    # Define a fallback Snowflake connector for when the real one isn't available
    class FallbackSnowflakeConnector:
        def check_existing_classification(self, domain):
            logger.info(f"Fallback: No existing classification for {domain}")
            return None
            
        def save_domain_content(self, domain, url, content):
            logger.info(f"Fallback: Not saving domain content for {domain}")
            return True, None
            
        def save_classification(self, domain, company_type, confidence_score, all_scores, model_metadata, low_confidence, detection_method):
            logger.info(f"Fallback: Not saving classification for {domain}")
            return True, None
    
    snowflake_conn = FallbackSnowflakeConnector()

# Simple in-memory storage
# This is just for the current request
job_status = {}
job_results = {}

def start_apify_crawl(url):
    """Start a crawl of the specified URL using Apify."""
    try:
        endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
        payload = {
            "startUrls": [{"url": url}],
            "maxCrawlingDepth": 1,  # Limit depth for faster crawls
            "maxCrawlPages": 10,    # Limit pages for faster crawls
            "timeout": 300          # 5 minute timeout
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['data']['id']
    except Exception as e:
        logger.error(f"Error starting Apify crawl: {e}")
        raise ValueError(f"Failed to start crawl: {e}")

def fetch_apify_results(run_id, timeout=300, interval=10):
    """Fetch the results of an Apify crawl."""
    try:
        endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_API_TOKEN}"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()
            
            if data:
                combined_text = ' '.join(item.get('text', '') for item in data if item.get('text'))
                domain_url = data[0].get('url', '') if data else ''
                return {
                    'success': True,
                    'domain': domain_url,
                    'content': combined_text,
                    'pages_crawled': len(data)
                }
            time.sleep(interval)
        
        return {'success': False, 'error': 'Timeout or no data returned'}
    except Exception as e:
        logger.error(f"Error fetching Apify results: {e}")
        return {'success': False, 'error': str(e)}

# Helper function to ensure all values are JSON serializable
def ensure_serializable(obj):
    if isinstance(obj, dict):
        return {k: ensure_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def background_crawl_and_classify(job_id, url):
    """Run the crawl and classify process in the background"""
    try:
        # Update status
        job_status[job_id] = "pending"
        
        # Parse domain from URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = f"https://{url}"
        domain = parsed_url.netloc or url
        
        # Check existing records in Snowflake first
        existing_record = snowflake_conn.check_existing_classification(domain)
        if existing_record:
            logger.info(f"Background job {job_id}: Found existing record for {domain}")
            
            # Try to parse all_scores if it exists
            confidence_scores = {}
            try:
                if 'all_scores' in existing_record:
                    confidence_scores = json.loads(existing_record.get('all_scores', '{}'))
            except Exception as e:
                logger.warning(f"Could not parse all_scores for {domain}: {e}")
                
            # Try to extract LLM explanation
            llm_explanation = ""
            try:
                metadata = json.loads(existing_record.get('model_metadata', '{}'))
                llm_explanation = metadata.get('llm_explanation', '')
            except Exception as e:
                logger.warning(f"Could not parse model_metadata for {domain}: {e}")
            
            # Generate a reasoning text based on available data
            if llm_explanation:
                reasoning = llm_explanation
            else:
                reasoning = f"Classification based on previously analyzed data. Detection method: {existing_record.get('detection_method', 'unknown')}"
            
            # Prepare result
            result = {
                "domain": domain,
                "predicted_class": existing_record.get('company_type', 'Unknown'),
                "confidence_score": existing_record.get('confidence_score', 0),
                "confidence_scores": confidence_scores,
                "low_confidence": existing_record.get('low_confidence', True),
                "detection_method": existing_record.get('detection_method', 'unknown'),
                "reasoning": reasoning,
                "source": "cached"
            }
            
            # Store result in memory
            job_results[job_id] = result
            job_status[job_id] = "completed"
            return
        
        # Start the crawl
        logger.info(f"Background job {job_id}: Starting crawl for {url}")
        crawl_run_id = start_apify_crawl(url)
        
        # Update status
        job_status[job_id] = "crawling"
        
        # Wait for crawl to complete
        while True:
            crawl_results = fetch_apify_results(crawl_run_id)
            
            if crawl_results.get('success'):
                break
                
            if crawl_results.get('error') and "Timeout" not in crawl_results.get('error'):
                job_status[job_id] = "failed"
                job_results[job_id] = {"error": crawl_results.get('error', 'Unknown error')}
                return
                
            time.sleep(5)
        
        # Process the crawl results
        content = crawl_results.get('content', '')
        logger.info(f"Background job {job_id}: Crawl completed for {domain}, got {len(content)} characters of content")
        
        # Update status
        job_status[job_id] = "classifying"
        
        # Classify the domain
        classification = classifier.classify_domain(content, domain=domain)
        classification = ensure_serializable(classification)
        
        # Fix for missing max_confidence field
        if 'max_confidence' not in classification:
            confidence_scores = classification.get('confidence_scores', {})
            max_confidence = max(confidence_scores.values()) if confidence_scores else 0.5
            classification['max_confidence'] = max_confidence
        
        # Generate reasoning
        reasoning = classification.get('llm_explanation', '')
        if not reasoning:
            # Fallback reasoning based on confidence scores
            scores = classification.get('confidence_scores', {})
            if scores:
                reasoning = f"Classification determined by confidence scores: "
                for cls, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    reasoning += f"{cls}: {score:.2f}, "
            else:
                reasoning = f"Classification method: {classification.get('detection_method', 'unknown')}"
        
        # Save to Snowflake
        try:
            save_to_snowflake(domain, url, content, classification)
        except Exception as e:
            logger.error(f"Background job {job_id}: Error saving to Snowflake: {e}")
        
        # Prepare result
        result = {
            "domain": domain,
            "predicted_class": str(classification['predicted_class']),
            "confidence_score": float(classification['max_confidence']),
            "confidence_scores": classification.get('confidence_scores', {}),
            "low_confidence": bool(classification.get('low_confidence', False)),
            "detection_method": str(classification.get('detection_method', 'unknown')),
            "reasoning": reasoning,
            "source": "fresh"
        }
        
        # Store result in memory
        job_results[job_id] = result
        job_status[job_id] = "completed"
        logger.info(f"Background job {job_id}: Classification completed")
        
    except Exception as e:
        logger.exception(f"Background job {job_id}: Error in background process: {str(e)}")
        job_status[job_id] = "failed"
        job_results[job_id] = {"error": str(e)}

@app.route('/classify-domain', methods=['POST', 'OPTIONS'])
def classify_domain():
    """Synchronous endpoint that processes the domain classification immediately"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    url = data.get('url')
    force_reclassify = data.get('force_reclassify', False)  # New parameter to force reclassification
    
    logger.info(f"Received request for {url}, force_reclassify={force_reclassify}")
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        # Parse domain from URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = f"https://{url}"
        domain = parsed_url.netloc or url
        
        # Check for existing content in Snowflake first
        existing_content = None
        existing_classification = None
        
        if hasattr(snowflake_conn, 'connected') and snowflake_conn.connected:
            # Try to get existing content
            conn = snowflake_conn.get_connection()
            if conn:
                cursor = conn.cursor()
                try:
                    cursor.execute("""
                        SELECT domain, url, text_content, crawl_date 
                        FROM DOMAIN_CONTENT 
                        WHERE domain = %s 
                        ORDER BY crawl_date DESC LIMIT 1
                    """, (domain,))
                    content_row = cursor.fetchone()
                    
                    if content_row:
                        existing_content = {
                            'domain': content_row[0],
                            'url': content_row[1],
                            'content': content_row[2],
                            'crawl_date': content_row[3]
                        }
                        logger.info(f"Found existing content for {domain} from {existing_content['crawl_date']}")
                        
                    # Check if we have a previous classification
                    if not force_reclassify:
                        # Only check for existing classification if not forcing reclassification
                        existing_classification = snowflake_conn.check_existing_classification(domain)
                        if existing_classification:
                            logger.info(f"Found existing classification for {domain}")
                except Exception as e:
                    logger.error(f"Error querying existing content: {e}")
                finally:
                    cursor.close()
                    conn.close()
        
        # If we have an existing classification and not forcing reclassify, return it
        if existing_classification and not force_reclassify:
            # Try to parse all_scores if it exists
            confidence_scores = {}
            try:
                if 'all_scores' in existing_classification:
                    confidence_scores = json.loads(existing_classification.get('all_scores', '{}'))
            except Exception as e:
                logger.warning(f"Could not parse all_scores for {domain}: {e}")
                
            # Try to extract LLM explanation
            llm_explanation = ""
            try:
                metadata = json.loads(existing_classification.get('model_metadata', '{}'))
                llm_explanation = metadata.get('llm_explanation', '')
            except Exception as e:
                logger.warning(f"Could not parse model_metadata for {domain}: {e}")
            
            # Generate a reasoning text based on available data
            if llm_explanation:
                reasoning = llm_explanation
            else:
                reasoning = f"Classification based on previously analyzed data. Detection method: {existing_classification.get('detection_method', 'unknown')}"
            
            return jsonify({
                "domain": domain,
                "predicted_class": existing_classification.get('company_type', 'Unknown'),
                "confidence_score": existing_classification.get('confidence_score', 0),
                "confidence_scores": confidence_scores,
                "low_confidence": existing_classification.get('low_confidence', True),
                "detection_method": existing_classification.get('detection_method', 'unknown'),
                "reasoning": reasoning,
                "source": "cached" 
            }), 200
        
        # If we have existing content, use it for classification (reclassify flow)
        content = None
        if existing_content:
            logger.info(f"Using existing content for {domain}")
            content = existing_content['content']
            # This is where we're reclassifying using existing content
        
        # Otherwise, crawl the site
        if not content:
            logger.info(f"Starting crawl for {url}")
            crawl_run_id = start_apify_crawl(url)
            
            # Wait for crawl to complete
            while True:
                crawl_results = fetch_apify_results(crawl_run_id)
                
                if crawl_results.get('success'):
                    break
                    
                if crawl_results.get('error') and "Timeout" not in crawl_results.get('error'):
                    return jsonify({"error": crawl_results.get('error', 'Unknown error')}), 500
                    
                time.sleep(5)
            
            # Process the crawl results
            content = crawl_results.get('content', '')
            logger.info(f"Crawl completed for {domain}, got {len(content)} characters of content")
            
            # Save content to Snowflake if it's new
            if not existing_content:
                try:
                    save_result, error = snowflake_conn.save_domain_content(domain=domain, url=url, content=content)
                    if save_result:
                        logger.info(f"Content saved to Snowflake for {domain}")
                    else:
                        logger.error(f"Failed to save content to Snowflake: {error}")
                except Exception as e:
                    logger.error(f"Error saving content to Snowflake: {e}")
        
        # Classify the domain
        logger.info(f"Classifying content for {domain}, content length: {len(content)}")
        classification = classifier.classify_domain(content, domain=domain)
        classification = ensure_serializable(classification)
        
        # Print classification details for debugging
        logger.info(f"Raw classification result: {json.dumps(classification)}")
        
        # Fix for missing max_confidence field
        if 'max_confidence' not in classification:
            confidence_scores = classification.get('confidence_scores', {})
            max_confidence = max(confidence_scores.values()) if confidence_scores else 0.5
            classification['max_confidence'] = max_confidence
        
        # Generate reasoning
        reasoning = classification.get('llm_explanation', '')
        if not reasoning:
            # Fallback reasoning based on confidence scores
            scores = classification.get('confidence_scores', {})
            if scores:
                reasoning = f"Classification determined by confidence scores: "
                for cls, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                    reasoning += f"{cls}: {score:.2f}, "
            else:
                reasoning = f"Classification method: {classification.get('detection_method', 'unknown')}"
        
        # Save to Snowflake
        try:
            logger.info(f"Saving classification to Snowflake for {domain}: {classification['predicted_class']}")
            save_result = save_to_snowflake(domain, url, content, classification)
            if save_result:
                logger.info(f"Classification saved to Snowflake for {domain}")
            else:
                logger.error(f"Failed to save classification to Snowflake for {domain}")
        except Exception as e:
            logger.error(f"Error saving to Snowflake: {e}")
        
        # Verify predicted_class matches confidence scores
        confidence_scores = classification.get('confidence_scores', {})
        if confidence_scores and classification['predicted_class'] != max(confidence_scores.items(), key=lambda x: x[1])[0]:
            logger.warning(f"Predicted class doesn't match highest confidence score! predicted={classification['predicted_class']}, highest={max(confidence_scores.items(), key=lambda x: x[1])}")
        
        # Return the result
        return jsonify({
            "domain": domain,
            "predicted_class": str(classification['predicted_class']),
            "confidence_score": float(classification['max_confidence']),
            "confidence_scores": classification.get('confidence_scores', {}),
            "low_confidence": bool(classification.get('low_confidence', False)),
            "detection_method": str(classification.get('detection_method', 'unknown')),
            "reasoning": reasoning,
            "source": "fresh" if not existing_content else "reclassified"
        }), 200
        
    except Exception as e:
        logger.exception(f"Error in classify-domain: {str(e)}")
        return jsonify({"error": str(e)}), 500

def save_to_snowflake(domain, url, content, classification):
    """Helper function to save classification data to Snowflake"""
    try:
        # Ensure domain content is saved
        success_content, error = snowflake_conn.save_domain_content(
            domain=domain, url=url, content=content
        )
        
        if not success_content:
            logger.error(f"Failed to save domain content: {error}")

        # Ensure max_confidence exists
        if 'max_confidence' not in classification:
            confidence_scores = classification.get('confidence_scores', {})
            max_confidence = max(confidence_scores.values()) if confidence_scores else 0.5
            classification['max_confidence'] = max_confidence

        llm_explanation = classification.get('llm_explanation', '')
        model_metadata = {
            'model_version': '1.0',
            'llm_model': 'claude-3-haiku-20240307',
            'llm_explanation': llm_explanation[:1000] if llm_explanation else ''
        }
        
        logger.info(f"Saving classification to Snowflake: {domain}, {classification['predicted_class']}, {classification['max_confidence']}")

        success_class, error_class = snowflake_conn.save_classification(
            domain=domain,
            company_type=str(classification['predicted_class']),
            confidence_score=float(classification['max_confidence']),
            all_scores=json.dumps(classification.get('confidence_scores', {})),
            model_metadata=json.dumps(model_metadata),
            low_confidence=bool(classification.get('low_confidence', False)),
            detection_method=str(classification.get('detection_method', 'unknown'))
        )
        
        if not success_class:
            logger.error(f"Failed to save classification: {error_class}")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error saving to Snowflake: {e}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify the API is running"""
    return jsonify({
        "status": "ok",
        "numpy_version": np.__version__,
        "model_fallback": getattr(classifier, '_using_fallback', True),
        "llm_available": hasattr(classifier, 'llm_classifier') and classifier.llm_classifier is not None,
        "pinecone_available": hasattr(classifier, 'pinecone_index') and classifier.pinecone_index is not None,
        "snowflake_connected": getattr(snowflake_conn, 'connected', False)
    }), 200

@app.route('/force-reclassify', methods=['POST', 'OPTIONS'])
def force_reclassify():
    """Endpoint to force reclassification of a domain using existing content"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    # Call the classify_domain function with force_reclassify=True
    data['force_reclassify'] = True
    return classify_domain()
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5005)), debug=False, use_reloader=False)
