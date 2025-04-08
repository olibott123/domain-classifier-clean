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

# [Rest of your existing functions - start_apify_crawl, fetch_apify_results, ensure_serializable]

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
                "domain": domain,
                "predicted_class": existing_record.get('company_type', 'Unknown'),
                "confidence_score": existing_record.get('confidence_score', 0),
                "confidence_scores": confidence_scores,
                "low_confidence": existing_record.get('low_confidence', True),
                "detection_method": existing_record.get('detection_method', 'unknown'),
                "reasoning": reasoning,
                "source": "cached" 
            }), 200

        # If no existing record, proceed with crawl and classification
        logger.info(f"Starting crawl for {url}")
        crawl_run_id = start_apify_crawl(url)
        crawl_results = fetch_apify_results(crawl_run_id)

        if not crawl_results['success']:
            logger.error(f"Crawl failed: {crawl_results.get('error', 'Unknown error')}")
            return jsonify({"error": crawl_results.get('error', 'Unknown error')}), 500

        content = crawl_results['content']
        logger.info(f"Crawl completed for {domain}, got {len(content)} characters of content")
        
        # Classify the domain
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
        save_to_snowflake(domain, url, content, classification)
        
        # Return enhanced response with reasoning
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

# Keep your existing crawl-and-save endpoint

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, debug=True, use_reloader=False)
