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

# Create a directory for job storage
JOB_DIR = '/workspace/jobs'
os.makedirs(JOB_DIR, exist_ok=True)

def save_job_status(job_id, status, result=None):
    """Save job status and result to file"""
    try:
        job_data = {
            "status": status,
            "updated": time.time()
        }
        
        if result is not None:
            job_data["result"] = result
            
        with open(os.path.join(JOB_DIR, f"{job_id}.json"), 'w') as f:
            json.dump(job_data, f)
        
        logger.info(f"Saved job status for {job_id}: {status}")
        return True
    except Exception as e:
        logger.error(f"Error saving job status: {e}")
        return False

def get_job_status(job_id):
    """Get job status and result from file"""
    try:
        job_file = os.path.join(JOB_DIR, f"{job_id}.json")
        if os.path.exists(job_file):
            with open(job_file, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error getting job status: {e}")
        return None

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

def process_domain_async(job_id, url, force_reclassify=False):
    """Process the domain classification in a background thread"""
    try:
        # Update job status
        save_job_status(job_id, "processing")
        
        # Parse domain from URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = f"https://{url}"
        domain = parsed_url.netloc or url
        
        logger.info(f"Job {job_id}: Processing {domain}, force_reclassify={force_reclassify}")
        
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
                        logger.info(f"Job {job_id}: Found existing content for {domain}")
                        
                    # Check if we have a previous classification
                    if not force_reclassify:
                        # Only check for existing classification if not forcing reclassification
                        existing_classification = snowflake_conn.check_existing_classification(domain)
                        if existing_classification:
                            logger.info(f"Job {job_id}: Found existing classification for {domain}")
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
            
            result = {
                "domain": domain,
                "predicted_class": existing_classification.get('company_type', 'Unknown'),
                "confidence_score": existing_classification.get('confidence_score', 0),
                "confidence_scores": confidence_scores,
                "low_confidence": existing_classification.get('low_confidence', True),
                "detection_method": existing_classification.get('detection_method', 'unknown'),
                "reasoning": reasoning,
                "source": "cached" 
            }
            
            save_job_status(job_id, "completed", result)
            return
        
        # If we have existing content, use it for classification
        content = None
        if existing_content:
            logger.info(f"Job {job_id}: Using existing content for {domain}")
            content = existing_content['content']
            save_job_status(job_id, "classifying")
        else:
            # Otherwise, crawl the site
            logger.info(f"Job {job_id}: Starting crawl for {url}")
            save_job_status(job_id, "crawling")
            
            try:
                crawl_run_id = start_apify_crawl(url)
                
                # Wait for crawl to complete
                crawl_results = None
                for _ in range(30):  # try for up to 5 minutes (30 * 10 seconds)
                    crawl_results = fetch_apify_results(crawl_run_id)
                    
                    if crawl_results.get('success'):
                        break
                        
                    if crawl_results.get('error') and "Timeout" not in crawl_results.get('error'):
                        save_job_status(job_id, "failed", {"error": crawl_results.get('error', 'Unknown error')})
                        return
                        
                    time.sleep(10)
                
                if not crawl_results or not crawl_results.get('success'):
                    save_job_status(job_id, "failed", {"error": "Crawl timeout or no data returned"})
                    return
                
                # Process the crawl results
                content = crawl_results.get('content', '')
                logger.info(f"Job {job_id}: Crawl completed for {domain}, got {len(content)} characters")
                
                # Save content to Snowflake
                try:
                    save_result, error = snowflake_conn.save_domain_content(domain=domain, url=url, content=content)
                    if save_result:
                        logger.info(f"Job {job_id}: Content saved to Snowflake for {domain}")
                    else:
                        logger.error(f"Job {job_id}: Failed to save content to Snowflake: {error}")
                except Exception as e:
                    logger.error(f"Job {job_id}: Error saving content to Snowflake: {e}")
                
                save_job_status(job_id, "classifying")
            except Exception as e:
                logger.error(f"Job {job_id}: Error during crawl: {e}")
                save_job_status(job_id, "failed", {"error": str(e)})
                return
        
        # Classify the domain
        try:
            logger.info(f"Job {job_id}: Classifying content for {domain}")
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
                logger.info(f"Job {job_id}: Saving classification to Snowflake for {domain}")
                save_result = save_to_snowflake(domain, url, content, classification)
                if save_result:
                    logger.info(f"Job {job_id}: Classification saved to Snowflake for {domain}")
                else:
                    logger.error(f"Job {job_id}: Failed to save classification to Snowflake")
            except Exception as e:
                logger.error(f"Job {job_id}: Error saving to Snowflake: {e}")
            
            # Prepare result
            result = {
                "domain": domain,
                "predicted_class": str(classification['predicted_class']),
                "confidence_score": float(classification['max_confidence']),
                "confidence_scores": classification.get('confidence_scores', {}),
                "low_confidence": bool(classification.get('low_confidence', False)),
                "detection_method": str(classification.get('detection_method', 'unknown')),
                "reasoning": reasoning,
                "source": "fresh" if not existing_content else "reclassified"
            }
            
            save_job_status(job_id, "completed", result)
            logger.info(f"Job {job_id}: Classification completed for {domain}")
        except Exception as e:
            logger.error(f"Job {job_id}: Error during classification: {e}")
            save_job_status(job_id, "failed", {"error": str(e)})
    except Exception as e:
        logger.exception(f"Job {job_id}: Unexpected error: {e}")
        save_job_status(job_id, "failed", {"error": str(e)})

@app.route('/start-classification', methods=['POST', 'OPTIONS'])
def start_classification():
    """Start a domain classification job and return immediately with a job ID"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
        
    data = request.get_json()
    url = data.get('url')
    force_reclassify = data.get('force_reclassify', False)
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    # Generate a job ID
    job_id = str(uuid.uuid4())
    
    # Start a background thread to handle the classification
    thread = threading.Thread(
        target=process_domain_async, 
        args=(job_id, url, force_reclassify)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "job_id": job_id,
        "status": "processing",
        "message": "Classification started, check status with /check-classification/{job_id}"
    }), 202

@app.route('/check-classification/<job_id>', methods=['GET'])
def check_classification(job_id):
    """Check the status of a classification job"""
    job_data = get_job_status(job_id)
    
    if not job_data:
        return jsonify({"error": "Job not found"}), 404
    
    status = job_data.get('status')
    
    if status == "completed":
        # Return the completed result
        return jsonify(job_data.get('result')), 200
    
    if status == "failed":
        # Return error information
        return jsonify({"error": "Job failed", "details": job_data.get('result', {})}), 500
    
    # Job is still in progress
    return jsonify({"status": status, "message": f"Job is {status}"}), 202

@app.route('/classify-domain', methods=['POST', 'OPTIONS'])
def classify_domain():
    """Legacy endpoint that now uses the async processing system but pretends to be synchronous"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    url = data.get('url')
    force_reclassify = data.get('force_reclassify', False)
    
    logger.info(f"Received request for {url}, force_reclassify={force_reclassify}")
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    # Parse domain from URL
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = f"https://{url}"
    domain = parsed_url.netloc or url
    
    # Check if we have an existing classification (quick response)
    if not force_reclassify and hasattr(snowflake_conn, 'connected') and snowflake_conn.connected:
        existing_classification = snowflake_conn.check_existing_classification(domain)
        if existing_classification:
            logger.info(f"Found existing classification for {domain}")
            
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
    
    # Otherwise, start async processing and return a job ID
    # Generate a job ID
    job_id = str(uuid.uuid4())
    
    # Start a background thread to handle the classification
    thread = threading.Thread(
        target=process_domain_async, 
        args=(job_id, url, force_reclassify)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "job_id": job_id,
        "status": "processing",
        "message": "Classification started, check status with /check-classification/{job_id}"
    }), 202

@app.route('/force-reclassify', methods=['POST', 'OPTIONS'])
def force_reclassify():
    """Endpoint to force reclassification of a domain"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    # Start async processing with force_reclassify=True
    job_id = str(uuid.uuid4())
    
    # Start a background thread to handle the classification
    thread = threading.Thread(
        target=process_domain_async, 
        args=(job_id, url, True)  # force_reclassify=True
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        "job_id": job_id,
        "status": "processing",
        "message": "Reclassification started, check status with /check-classification/{job_id}"
    }), 202

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
        "snowflake_connected": getattr(snowflake_conn, 'connected', False),
        "job_dir_exists": os.path.exists(JOB_DIR)
    }), 200
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5005)), debug=False, use_reloader=False)
