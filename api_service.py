from flask import Flask, request, jsonify
from flask_cors import CORS
from domain_classifier import DomainClassifier
from snowflake_connector import SnowflakeConnector
import requests
import time
from urllib.parse import urlparse
import json
import os
import numpy as np
import logging

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

# Get API keys and settings from environment variables
APIFY_TASK_ID = os.environ.get("APIFY_TASK_ID", "z3plE6RoQ5W6SNLDe")
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
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

snowflake_conn = SnowflakeConnector()

def start_apify_crawl(url):
    """Start a crawl of the specified URL using Apify."""
    try:
        endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
        payload = {
            "startUrls": [{"url": url}],
            "maxCrawlingDepth": 1,  # Limit depth for faster crawls
            "maxCrawlPages": 10     # Limit pages for faster crawls
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()['data']['id']
    except Exception as e:
        logger.error(f"Error starting Apify crawl: {e}")
        raise ValueError(f"Failed to start crawl: {e}")

def fetch_apify_results(run_id, timeout=5, interval=2):
    """
    Fetch the results of an Apify crawl.
    The short timeout is used for status checks, not for the full crawl.
    """
    try:
        endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_API_TOKEN}"
        status_endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}?token={APIFY_API_TOKEN}"
        
        # First check if the run is still in progress
        status_response = requests.get(status_endpoint)
        status_response.raise_for_status()
        status_data = status_response.json()
        
        status = status_data.get('status')
        
        if status in ['RUNNING', 'READY']:
            return {'success': False, 'error': 'Crawl in progress', 'status': status}
        
        if status in ['FAILED', 'ABORTED', 'TIMED-OUT']:
            return {'success': False, 'error': f'Crawl {status}', 'status': status}
        
        # If the run is finished, get the data
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()
            
            if data:
                combined_text = ' '.join(item['text'] for item in data if item.get('text'))
                domain_url = data[0].get('url', '')
                return {
                    'success': True,
                    'domain': domain_url,
                    'content': combined_text,
                    'pages_crawled': len(data),
                    'status': 'SUCCEEDED'
                }
            time.sleep(interval)
        
        return {'success': False, 'error': 'No data returned', 'status': status}
    except Exception as e:
        logger.error(f"Error fetching Apify results: {e}")
        return {'success': False, 'error': str(e), 'status': 'ERROR'}

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

@app.route('/start-crawl', methods=['POST', 'OPTIONS'])
def start_crawl():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
        
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        # Ensure URL is properly formatted
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = f"https://{url}"
        
        domain = parsed_url.netloc or url  # Fallback to original if parsing fails

        # Check existing records in Snowflake first
        existing_record = snowflake_conn.check_existing_classification(domain)
        if existing_record:
            logger.info(f"[Snowflake] Existing record found for domain {domain}")
            
            # Try to parse all_scores if it exists
            confidence_scores = {}
            try:
                if 'all_scores' in existing_record:
                    confidence_scores = json.loads(existing_record.get('all_scores', '{}'))
            except:
                logger.warning(f"Could not parse all_scores for {domain}")
                
            # Try to extract LLM explanation
            llm_explanation = ""
            try:
                metadata = json.loads(existing_record.get('model_metadata', '{}'))
                llm_explanation = metadata.get('llm_explanation', '')
            except:
                logger.warning(f"Could not parse model_metadata for {domain}")
            
            # Generate a reasoning text based on available data
            if llm_explanation:
                reasoning = llm_explanation
            else:
                reasoning = f"Classification based on previously analyzed data. Detection method: {existing_record.get('detection_method', 'unknown')}"
            
            return jsonify({
                "status": "complete",
                "domain": domain,
                "predicted_class": existing_record.get('company_type', 'Unknown'),
                "confidence_score": existing_record.get('confidence_score', 0),
                "confidence_scores": confidence_scores,
                "low_confidence": existing_record.get('low_confidence', True),
                "detection_method": existing_record.get('detection_method', 'unknown'),
                "reasoning": reasoning,
                "source": "cached" 
            }), 200

        # If no existing record, start a new crawl
        logger.info(f"Starting new crawl for {url}")
        crawl_run_id = start_apify_crawl(url)
        
        return jsonify({
            "status": "started",
            "domain": domain,
            "crawl_id": crawl_run_id,
            "message": "Classification started. Check status with /check-crawl endpoint."
        }), 202  # 202 Accepted

    except Exception as e:
        logger.exception(f"Error in start-crawl: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/check-crawl/<crawl_id>', methods=['GET'])
def check_crawl(crawl_id):
    try:
        # Check if the crawl is complete
        crawl_results = fetch_apify_results(crawl_id)
        
        # If still in progress
        if not crawl_results.get('success'):
            if crawl_results.get('status') in ['RUNNING', 'READY']:
                # Still running
                return jsonify({
                    "status": "in_progress",
                    "message": "Crawl is still in progress. Please check back in a few seconds."
                }), 200
            else:
                # Failed for some reason
                return jsonify({
                    "status": "failed",
                    "error": crawl_results.get('error', 'Unknown error')
                }), 500
        
        # Crawl finished - process the results
        domain = urlparse(crawl_results.get('domain', '')).netloc
        content = crawl_results.get('content', '')
        
        if not content or len(content.split()) < 20:
            return jsonify({
                "status": "failed",
                "error": "Not enough content crawled to classify domain."
            }), 400
        
        # Classify domain
        classification = classifier.classify_domain(content, domain=domain)
        classification = ensure_serializable(classification)
        
        # Generate a detailed reasoning text
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
        save_to_snowflake(domain, crawl_results.get('domain', ''), content, classification)
        
        # Return enhanced response with reasoning
        return jsonify({
            "status": "complete",
            "domain": domain,
            "predicted_class": str(classification['predicted_class']),
            "confidence_score": float(classification['max_confidence']),
            "confidence_scores": classification.get('confidence_scores', {}),
            "low_confidence": bool(classification['low_confidence']),
            "detection_method": str(classification['detection_method']),
            "reasoning": reasoning,
            "source": "fresh",
            "pages_crawled": crawl_results.get('pages_crawled', 0)
        }), 200

    except Exception as e:
        logger.exception(f"Error in check-crawl: {str(e)}")
        return jsonify({"status": "error", "error": str(e)}), 500

# Maintain the original classify-domain endpoint for backward compatibility
@app.route('/classify-domain', methods=['POST', 'OPTIONS'])
def classify_domain():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
        
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        # First check if we have a cached result
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = f"https://{url}"
        
        domain = parsed_url.netloc or url
        existing_record = snowflake_conn.check_existing_classification(domain)
        
        if existing_record:
            # Return cached result with the same format as before
            logger.info(f"[Snowflake] Returning cached result for {domain}")
            
            # Extract fields from existing record
            confidence_scores = {}
            try:
                if 'all_scores' in existing_record:
                    confidence_scores = json.loads(existing_record.get('all_scores', '{}'))
            except:
                logger.warning(f"Could not parse all_scores for {domain}")
                
            llm_explanation = ""
            try:
                metadata = json.loads(existing_record.get('model_metadata', '{}'))
                llm_explanation = metadata.get('llm_explanation', '')
            except:
                logger.warning(f"Could not parse model_metadata for {domain}")
            
            if llm_explanation:
                reasoning = llm_explanation
            else:
                reasoning = f"Classification based on previously analyzed data. Detection method: {existing_record.get('detection_method', 'unknown')}"
            
            return jsonify({
                "domain": domain,
                "predicted_class": existing_record.get('company_type', 'Unknown'),
                "confidence_score": existing_record.get('confidence_score', 0),
                "confidence_scores": confidence_scores,
                "low_confidence": existing_record.get('low_confidence', True),
                "detection_method": existing_record.get('detection_method', 'unknown'),
                "reasoning": reasoning,
                "source": "cached" 
            }), 200
        
        # If no cached result, start a crawl and wait for completion 
        crawl_run_id = start_apify_crawl(url)
        
        # Poll for results with a maximum wait time of 60 seconds
        max_wait_time = 60  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            crawl_results = fetch_apify_results(crawl_run_id)
            
            if crawl_results.get('success'):
                # Crawl completed successfully
                content = crawl_results.get('content', '')
                
                # Classify the domain
                classification = classifier.classify_domain(content, domain=domain)
                classification = ensure_serializable(classification)
                
                # Generate reasoning
                reasoning = classification.get('llm_explanation', '')
                if not reasoning:
                    scores = classification.get('confidence_scores', {})
                    if scores:
                        reasoning = f"Classification determined by confidence scores: "
                        for cls, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
                            reasoning += f"{cls}: {score:.2f}, "
                    else:
                        reasoning = f"Classification method: {classification.get('detection_method', 'unknown')}"
                
                # Save to Snowflake
                save_to_snowflake(domain, url, content, classification)
                
                # Return the result in the original format
                return jsonify({
                    "domain": domain,
                    "predicted_class": str(classification['predicted_class']),
                    "confidence_score": float(classification['max_confidence']),
                    "confidence_scores": classification.get('confidence_scores', {}),
                    "low_confidence": bool(classification['low_confidence']),
                    "detection_method": str(classification['detection_method']),
                    "reasoning": reasoning,
                    "source": "fresh"
                }), 200
            
            elif crawl_results.get('status') in ['FAILED', 'ABORTED', 'TIMED-OUT', 'ERROR']:
                # Crawl failed
                return jsonify({
                    "error": f"Crawl failed: {crawl_results.get('error', 'Unknown error')}"
                }), 500
            
            # Still processing, wait and try again
            time.sleep(5)
        
        # If we get here, we've timed out
        return jsonify({
            "error": "Classification timed out. Please try again or use the two-step process with /start-crawl and /check-crawl endpoints."
        }), 504  # Gateway Timeout
        
    except Exception as e:
        logger.exception(f"Error in classify-domain: {str(e)}")
        return jsonify({"error": str(e)}), 500

def save_to_snowflake(domain, url, content, classification):
    """Helper function to save classification data to Snowflake"""
    try:
        success_content, _ = snowflake_conn.save_domain_content(
            domain=domain, url=url, content=content
        )

        llm_explanation = classification.get('llm_explanation', '')
        model_metadata = {
            'model_version': '1.0',
            'llm_model': 'claude-3-haiku-20240307',
            'llm_explanation': llm_explanation[:1000] if llm_explanation else ''
        }

        snowflake_conn.save_classification(
            domain=domain,
            company_type=str(classification['predicted_class']),
            confidence_score=float(classification['max_confidence']),
            all_scores=json.dumps(classification['confidence_scores']),
            model_metadata=json.dumps(model_metadata),
            low_confidence=bool(classification['low_confidence']),
            detection_method=str(classification['detection_method'])
        )
        return True
    except Exception as e:
        logger.error(f"Error saving to Snowflake: {e}")
        return False

@app.route('/crawl-and-save', methods=['POST'])
def crawl_and_save():
    data = request.get_json()
    url = data.get('url')
    if not url:
        return jsonify({"error": "URL or domain required"}), 400
    
    parsed_url = urlparse(url)
    if not parsed_url.scheme:
        url = f"https://{url}"
    domain = urlparse(url).netloc
    
    # Check existing records in Snowflake
    existing_record = snowflake_conn.check_existing_classification(domain)
    if existing_record:
        logger.info(f"[Snowflake] Existing record found for domain {domain}")
        return jsonify({"status": "exists", "domain": domain, "record": existing_record}), 200
    
    # No existing record, proceed to crawl
    try:
        logger.info(f"Starting crawl for {url}")
        crawl_run_id = start_apify_crawl(url)
        
        # Wait for crawl to complete with a longer timeout
        max_wait_time = 120  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            crawl_results = fetch_apify_results(crawl_run_id)
            
            if crawl_results.get('success'):
                break
                
            if crawl_results.get('status') in ['FAILED', 'ABORTED', 'TIMED-OUT', 'ERROR']:
                return jsonify({"error": crawl_results.get('error', 'Unknown error')}), 500
                
            time.sleep(5)
            
        if not crawl_results.get('success'):
            return jsonify({"error": "Crawl timed out"}), 504
        
        content = crawl_results['content']
        logger.info(f"Crawl completed for {domain}, got {len(content)} characters of content")
        
        # Classify the domain
        classification = classifier.classify_domain(content, domain=domain)
        classification = ensure_serializable(classification)
        
        # Saving to Snowflake
        logger.info(f"Saving domain content to Snowflake for {domain}")
        success_content, error_content = snowflake_conn.save_domain_content(
            domain=domain, url=url, content=content
        )
        
        # Store the additional LLM explanation in the metadata
        llm_explanation = classification.get('llm_explanation', '')
        model_metadata = {
            'model_version': '1.0',
            'llm_model': 'claude-3-haiku-20240307',
            'llm_explanation': llm_explanation[:1000] if llm_explanation else ''
        }
        
        logger.info(f"Sending classification to Snowflake: {classification['predicted_class']}")
        success_class, error_class = snowflake_conn.save_classification(
            domain=domain,
            company_type=str(classification['predicted_class']),
            confidence_score=float(classification['max_confidence']),
            all_scores=json.dumps(classification['confidence_scores']),
            model_metadata=json.dumps(model_metadata),
            low_confidence=bool(classification['low_confidence']),
            detection_method=str(classification['detection_method'])
        )
        
        # Build and sanitize response to ensure JSON serialization
        response_data = {
            "domain": domain,
            "classification": {
                "predicted_class": str(classification['predicted_class']),
                "confidence_scores": {k: float(v) for k, v in classification['confidence_scores'].items()},
                "max_confidence": float(classification['max_confidence']),
                "low_confidence": bool(classification['low_confidence']),
                "detection_method": str(classification['detection_method']),
                "llm_explanation": str(classification.get('llm_explanation', ''))
            },
            "snowflake": {
                "content_saved": bool(success_content),
                "classification_saved": bool(success_class),
                "errors": {
                    "content_error": str(error_content) if error_content else None,
                    "classification_error": str(error_class) if error_class else None
                }
            }
        }
        
        return jsonify(response_data), 200
    
    except Exception as e:
        logger.exception(f"Error in crawl-and-save: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Add a test endpoint to test LLM classification directly
@app.route('/test-llm-classification', methods=['POST'])
def test_llm_classification():
    data = request.get_json()
    domain = data.get('domain')
    content = data.get('content')
    
    if not content:
        return jsonify({"error": "Content required for classification test"}), 400
    
    try:
        # Directly use the LLM classifier part
        llm_result = classifier.llm_classifier.classify(content, domain)
        # Ensure all values are JSON serializable
        llm_result = ensure_serializable(llm_result)
        
        return jsonify({
            "domain": domain,
            "llm_classification": llm_result
        }), 200
    
    except Exception as e:
        logger.exception(f"Error in test-llm-classification: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True, use_reloader=False)
