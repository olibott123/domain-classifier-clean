from flask import Flask, request, jsonify
from flask_cors import CORS
from llm_classifier import LLMClassifier
from snowflake_connector import SnowflakeConnector
import requests
import time
from urllib.parse import urlparse
import json
import os
import numpy as np
import logging
import traceback

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

# Configuration
LOW_CONFIDENCE_THRESHOLD = 0.7  # Threshold below which we consider a classification "low confidence"
AUTO_RECLASSIFY_THRESHOLD = 0.6  # Threshold below which we automatically reclassify

# Get API keys and settings from environment variables
APIFY_TASK_ID = os.environ.get("APIFY_TASK_ID")
APIFY_API_TOKEN = os.environ.get("APIFY_API_TOKEN")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Initialize LLM classifier directly 
try:
    llm_classifier = LLMClassifier(
        api_key=ANTHROPIC_API_KEY,
        model="claude-3-haiku-20240307"
    )
    logger.info(f"Initialized LLM classifier with model: claude-3-haiku-20240307")
except Exception as e:
    logger.error(f"Failed to initialize LLM classifier: {e}")
    llm_classifier = None

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
            
        def get_domain_content(self, domain):
            logger.info(f"Fallback: No content for {domain}")
            return None
    
    snowflake_conn = FallbackSnowflakeConnector()

def crawl_website(url):
    """Crawl a website using Apify."""
    try:
        logger.info(f"Starting crawl for {url}")
        
        # Start the crawl
        endpoint = f"https://api.apify.com/v2/actor-tasks/{APIFY_TASK_ID}/runs?token={APIFY_API_TOKEN}"
        payload = {
            "startUrls": [{"url": url}],
            "maxCrawlingDepth": 1,  # Limit depth for faster crawls
            "maxCrawlPages": 10     # Limit pages for faster crawls
        }
        headers = {"Content-Type": "application/json"}
        
        response = requests.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        run_id = response.json()['data']['id']
        
        # Wait for crawl to complete
        endpoint = f"https://api.apify.com/v2/actor-runs/{run_id}/dataset/items?token={APIFY_API_TOKEN}"
        
        max_attempts = 30  # Try for up to 5 minutes (30 * 10 seconds)
        for attempt in range(max_attempts):
            logger.info(f"Checking crawl results, attempt {attempt+1}/{max_attempts}")
            
            response = requests.get(endpoint)
            response.raise_for_status()
            data = response.json()
            
            if data:
                combined_text = ' '.join(item.get('text', '') for item in data if item.get('text'))
                logger.info(f"Crawl completed, got {len(combined_text)} characters of content")
                return combined_text
            
            if attempt < max_attempts - 1:
                time.sleep(10)
        
        logger.warning(f"Crawl timed out after {max_attempts} attempts")
        return None
    except Exception as e:
        logger.error(f"Error crawling website: {e}")
        return None

def save_to_snowflake(domain, url, content, classification):
    """Save classification data to Snowflake"""
    try:
        # Always save the domain content
        logger.info(f"Saving content to Snowflake for {domain}")
        snowflake_conn.save_domain_content(
            domain=domain, url=url, content=content
        )

        # Ensure max_confidence exists
        if 'max_confidence' not in classification:
            confidence_scores = classification.get('confidence_scores', {})
            max_confidence = max(confidence_scores.values()) if confidence_scores else 0.5
            classification['max_confidence'] = max_confidence

        # Set low_confidence flag based on confidence threshold
        if 'low_confidence' not in classification:
            classification['low_confidence'] = classification['max_confidence'] < LOW_CONFIDENCE_THRESHOLD

        # Create model metadata with TRUNCATED explanation to avoid Snowflake error
        llm_explanation = classification.get('llm_explanation', '')
        model_metadata = {
            'model_version': '1.0',
            'llm_model': 'claude-3-haiku-20240307',
            'llm_explanation': llm_explanation[:500] if llm_explanation else ''  # Truncate to 500 chars
        }
        
        logger.info(f"Saving classification to Snowflake: {domain}, {classification['predicted_class']}")
        snowflake_conn.save_classification(
            domain=domain,
            company_type=str(classification['predicted_class']),
            confidence_score=float(classification['max_confidence']),
            all_scores=json.dumps(classification.get('confidence_scores', {}))[:4000],  # Limit size
            model_metadata=json.dumps(model_metadata)[:4000],  # Limit size
            low_confidence=bool(classification.get('low_confidence', False)),
            detection_method=str(classification.get('detection_method', 'llm_classification'))
        )
        
        return True
    except Exception as e:
        logger.error(f"Error saving to Snowflake: {e}\n{traceback.format_exc()}")
        return False

def extract_domain_from_email(email):
    """Extract domain from an email address."""
    try:
        # Simple validation of email format
        if not email or '@' not in email:
            return None
            
        # Extract domain portion (after @)
        domain = email.split('@')[-1].strip().lower()
        
        # Basic validation of domain
        if not domain or '.' not in domain:
            return None
            
        return domain
    except Exception as e:
        logger.error(f"Error extracting domain from email: {e}")
        return None

@app.route('/classify-domain', methods=['POST', 'OPTIONS'])
def classify_domain():
    """Direct API that classifies a domain and returns the result"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        url = data.get('url', '').strip()
        force_reclassify = data.get('force_reclassify', False)
        
        if not url:
            return jsonify({"error": "URL is required"}), 400
            
        # Format URL properly
        if not url.startswith('http'):
            url = 'https://' + url
            
        # Extract domain
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        if not domain:
            domain = parsed_url.path
            
        # Remove www. if present
        if domain.startswith('www.'):
            domain = domain[4:]
            
        if not domain:
            return jsonify({"error": "Invalid URL"}), 400
            
        logger.info(f"Processing classification request for {domain}")
        
        # Check for existing classification if not forcing reclassification
        if not force_reclassify:
            existing_record = snowflake_conn.check_existing_classification(domain)
            if existing_record:
                # Check if confidence is below threshold for auto-reclassification
                confidence_score = existing_record.get('confidence_score', 1.0)
                if confidence_score < AUTO_RECLASSIFY_THRESHOLD:
                    logger.info(f"Auto-reclassifying {domain} due to low confidence score: {confidence_score}")
                    force_reclassify = True
                else:
                    logger.info(f"Found existing classification for {domain}")
                    
                    # Extract confidence scores
                    confidence_scores = {}
                    try:
                        confidence_scores = json.loads(existing_record.get('all_scores', '{}'))
                    except Exception as e:
                        logger.warning(f"Could not parse all_scores for {domain}: {e}")
                    
                    # Extract LLM explanation
                    llm_explanation = ""
                    try:
                        metadata = json.loads(existing_record.get('model_metadata', '{}'))
                        llm_explanation = metadata.get('llm_explanation', '')
                    except Exception as e:
                        logger.warning(f"Could not parse model_metadata for {domain}: {e}")
                    
                    # Add low_confidence flag based on confidence score
                    low_confidence = existing_record.get('low_confidence', confidence_score < LOW_CONFIDENCE_THRESHOLD)
                    
                    # Return the cached classification
                    return jsonify({
                        "domain": domain,
                        "company_type": existing_record.get('company_type'),
                        "confidence": existing_record.get('confidence_score'),
                        "confidence_scores": confidence_scores,
                        "explanation": llm_explanation,
                        "low_confidence": low_confidence,
                        "source": "cached"
                    }), 200
        
        # Try to get content (either from DB or by crawling)
        content = None
        
        # If reclassifying, try to get existing content first
        if force_reclassify:
            try:
                content = snowflake_conn.get_domain_content(domain)
                if content:
                    logger.info(f"Using existing content for {domain}")
            except (AttributeError, Exception) as e:
                logger.warning(f"Could not get existing content, will crawl instead: {e}")
                content = None
        
        # If no content yet, crawl the website
        if not content:
            logger.info(f"Crawling website for {domain}")
            content = crawl_website(url)
            
            if not content:
                return jsonify({
                    "domain": domain,
                    "error": "Failed to crawl website",
                    "company_type": "Unknown",
                    "low_confidence": True
                }), 500
        
        # Classify the content
        if not llm_classifier:
            return jsonify({
                "domain": domain, 
                "error": "LLM classifier is not available",
                "company_type": "Unknown",
                "low_confidence": True
            }), 500
            
        logger.info(f"Classifying content for {domain}")
        classification = llm_classifier.classify(content, domain)
        
        if not classification:
            return jsonify({
                "domain": domain,
                "error": "Classification failed",
                "company_type": "Unknown",
                "low_confidence": True
            }), 500
        
        # Ensure predicted_class matches highest confidence score
        if "confidence_scores" in classification:
            confidence_scores = classification["confidence_scores"]
            if confidence_scores:
                max_class = max(confidence_scores.items(), key=lambda x: x[1])
                classification["predicted_class"] = max_class[0]
                classification["max_confidence"] = max_class[1]
        
        # Set low_confidence flag based on threshold
        classification["low_confidence"] = classification.get("max_confidence", 0) < LOW_CONFIDENCE_THRESHOLD
        
        # Save to Snowflake (always save, even for reclassifications)
        save_to_snowflake(domain, url, content, classification)
        
        # Return the classification result
        return jsonify({
            "domain": domain,
            "company_type": classification.get('predicted_class'),
            "confidence": classification.get('max_confidence', 0.0),
            "confidence_scores": classification.get('confidence_scores', {}),
            "explanation": classification.get('llm_explanation', ''),
            "low_confidence": classification.get('low_confidence', False),
            "source": "fresh"
        }), 200
        
    except Exception as e:
        logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "company_type": "Error",
            "low_confidence": True
        }), 500

@app.route('/classify-email', methods=['POST', 'OPTIONS'])
def classify_email():
    """Classify a domain extracted from an email address"""
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        data = request.json
        email = data.get('email', '').strip()
        
        if not email:
            return jsonify({"error": "Email is required"}), 400
            
        # Extract domain from email
        domain = extract_domain_from_email(email)
        if not domain:
            return jsonify({"error": "Invalid email format or failed to extract domain"}), 400
        
        logger.info(f"Extracted domain '{domain}' from email '{email}'")
        
        # Create a new request data with the extracted domain
        domain_data = {
            "url": domain,
            "force_reclassify": data.get('force_reclassify', False)
        }
        
        # Store the original request
        original_json = request.json
        
        # Modify the request object
        request.json = domain_data
        
        # Call the classify_domain function
        response = classify_domain()
        
        # Restore the original request
        request.json = original_json
        
        # If the response is a tuple (response, status_code)
        if isinstance(response, tuple):
            response_obj, status_code = response
            
            # If successful, add the email to the response
            if status_code == 200:
                response_data = json.loads(response_obj.get_data(as_text=True))
                response_data['email'] = email
                return jsonify(response_data), 200
            
            # Otherwise, just return the original response
            return response
            
        # Handle case where response is not a tuple
        try:
            response_data = json.loads(response.get_data(as_text=True))
            response_data['email'] = email
            return jsonify(response_data), 200
        except:
            return response
        
    except Exception as e:
        logger.error(f"Error processing email classification request: {e}\n{traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "company_type": "Error",
            "low_confidence": True
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        "status": "ok", 
        "llm_available": llm_classifier is not None,
        "snowflake_connected": getattr(snowflake_conn, 'connected', False)
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
