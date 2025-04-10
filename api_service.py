from flask import Flask, request, jsonify
from flask_cors import CORS
from domain_classifier_fixed import DomainClassifier
from snowflake_connector import SnowflakeConnector
import requests
from urllib.parse import urlparse
import json
import os
import numpy as np
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Environment variables for API keys and settings
APIFY_TASK_ID = os.environ.get("APIFY_TASK_ID")
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "domain-embeddings")

# Initialize Classifier
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
snowflake_conn = SnowflakeConnector()

def start_apify_crawl(url):
    """Start a crawl of the specified URL using Apify."""
    try:
        endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
        payload = {
            "startUrls": [{"url": url}],
            "maxCrawlingDepth": 1,
            "maxCrawlPages": 10,
            "timeout": 300  # 5 minute timeout
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

@app.route('/classify-domain', methods=['POST', 'OPTIONS'])
def classify_domain():
    """Process domain classification immediately with forced crawl and reclassification"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    data = request.get_json()
    url = data.get('url')
    
    if not url:
        return jsonify({"error": "URL is required"}), 400
    
    try:
        # Parse domain from URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme:
            url = f"https://{url}"
        domain = parsed_url.netloc or url
        
        # Always start a new crawl
        logger.info(f"Starting crawl for {url}")
        crawl_run_id = start_apify_crawl(url)
        
        # Wait for crawl to complete
        crawl_results = fetch_apify_results(crawl_run_id)
        
        if not crawl_results.get('success'):
            return jsonify({
                "error": crawl_results.get('error', 'Crawl failed'),
                "status": "failed"
            }), 500
        
        # Get the crawled content
        content = crawl_results.get('content', '')
        logger.info(f"Crawl completed for {domain}, got {len(content)} characters of content")
        
        # Classify the domain
        classification = classifier.classify_domain(content, domain=domain)
        
        # Save content and classification to Snowflake
        try:
            # Save domain content
            snowflake_conn.save_domain_content(
                domain=domain, 
                url=url, 
                content=content
            )
            
            # Save classification
            snowflake_conn.save_classification(
                domain=domain,
                company_type=str(classification['predicted_class']),
                confidence_score=float(classification.get('max_confidence', 0)),
                all_scores=json.dumps(classification.get('confidence_scores', {})),
                model_metadata=json.dumps({
                    'model_version': '1.0',
                    'llm_model': 'claude-3-haiku-20240307',
                    'llm_explanation': classification.get('llm_explanation', '')[:1000]
                }),
                low_confidence=bool(classification.get('low_confidence', False)),
                detection_method=str(classification.get('detection_method', 'unknown'))
            )
        except Exception as e:
            logger.error(f"Error saving to Snowflake: {e}")
        
        # Return classification result
        return jsonify({
            "domain": domain,
            "predicted_class": str(classification['predicted_class']),
            "confidence_score": float(classification.get('max_confidence', 0)),
            "confidence_scores": classification.get('confidence_scores', {}),
            "low_confidence": bool(classification.get('low_confidence', False)),
            "detection_method": str(classification.get('detection_method', 'unknown')),
            "reasoning": classification.get('llm_explanation', ''),
            "source": "reclassified"
        }), 200
        
    except Exception as e:
        logger.exception(f"Classification error: {str(e)}")
        return jsonify({
            "error": str(e),
            "status": "failed"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok",
        "snowflake_connected": snowflake_conn.connected,
        "pinecone_available": hasattr(classifier, 'pinecone_index') and classifier.pinecone_index is not None,
        "llm_available": hasattr(classifier, 'llm_classifier') and classifier.llm_classifier is not None
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5005)), debug=False, use_reloader=False)
