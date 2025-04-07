from flask import Flask, request, jsonify
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
    llm_model="claude-3-haiku-20240307"  # Using the fastest model for better response times
)

snowflake_conn = SnowflakeConnector()

def start_apify_crawl(url):
    endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
    payload = {"startUrls": [{"url": url}]}
    headers = {"Content-Type": "application/json"}
    response = requests.post(endpoint, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()['data']['id']

def fetch_apify_results(run_id, timeout=300, interval=10):
    endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_API_TOKEN}"
    start_time = time.time()
    while time.time() - start_time < timeout:
        response = requests.get(endpoint)
        response.raise_for_status()
        data = response.json()
        if data:
            combined_text = ' '.join(item['text'] for item in data if item.get('text') is not None)
            domain_url = data[0].get('url', '')
            return {
                'success': True,
                'domain': domain_url,
                'content': combined_text,
                'pages_crawled': len(data)
            }
        time.sleep(interval)
    return {'success': False, 'error': 'Timeout or no data returned'}

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
        logger.info(f"[Snowflake] Existing record found for domain {domain}: {existing_record}")
        return jsonify({"status": "exists", "domain": domain, "record": existing_record}), 200

    # No existing record, proceed to crawl
    try:
        logger.info(f"Starting crawl for {url}")
        crawl_run_id = start_apify_crawl(url)
        crawl_results = fetch_apify_results(crawl_run_id)

        if not crawl_results['success']:
            logger.error(f"Crawl failed: {crawl_results.get('error', 'Unknown error')}")
            return jsonify({"error": crawl_results.get('error', 'Unknown error')}), 500

        content = crawl_results['content']
        logger.info(f"Crawl completed for {domain}, got {len(content)} characters of content")
        
        # This is where the magic happens - classify the domain using both traditional and LLM methods
        classification = classifier.classify_domain(content, domain=domain)
        
        # Explicitly convert all values to standard Python types
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
            'llm_explanation': llm_explanation[:1000] if llm_explanation else ''  # Truncate if too long
        }

        logger.info(f"Sending classification to Snowflake: {classification['predicted_class']} with confidence {classification['max_confidence']}")

        success_class, error_class = snowflake_conn.save_classification(
            domain=domain,
            company_type=str(classification['predicted_class']),
            confidence_score=float(classification['max_confidence']),
            all_scores=json.dumps(classification['confidence_scores']),
            model_metadata=json.dumps(model_metadata),
            low_confidence=bool(classification['low_confidence']),
            detection_method=str(classification['detection_method'])
        )

        if not success_class:
            logger.error(f"ERROR saving classification to Snowflake: {error_class}")
        else:
            logger.info(f"SUCCESS saving classification to Snowflake for domain: {domain}")

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

# Add a new endpoint to test LLM classification directly
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
