from flask import request, jsonify
import logging
import os
import traceback

# Set up logging
logger = logging.getLogger(__name__)

# Import domain utilities
from domain_classifier.utils.domain_utils import extract_domain_from_email
from domain_classifier.utils.error_handling import detect_error_type, create_error_result

# Import configuration
from domain_classifier.config.overrides import check_domain_override

# Import services
from domain_classifier.classifiers.llm_classifier import LLMClassifier
from domain_classifier.storage.snowflake_connector import SnowflakeConnector
from domain_classifier.crawlers.apify_crawler import crawl_website
from domain_classifier.storage.operations import save_to_snowflake
from domain_classifier.classifiers.result_validator import validate_result_consistency

# Initialize services
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
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
    from domain_classifier.storage.fallback_connector import FallbackSnowflakeConnector
    snowflake_conn = FallbackSnowflakeConnector()

def register_routes(app):
    """Register all routes with the app."""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Simple health check endpoint"""
        return jsonify({
            "status": "ok", 
            "llm_available": llm_classifier is not None,
            "snowflake_connected": getattr(snowflake_conn, 'connected', False)
        }), 200
    
    @app.route('/classify-domain', methods=['POST', 'OPTIONS'])
    def classify_domain():
        """Direct API that classifies a domain or email and returns the result"""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            data = request.json
            input_value = data.get('url', '').strip()
            
            # Change default for force_reclassify to True
            force_reclassify = data.get('force_reclassify', True)
            
            use_existing_content = data.get('use_existing_content', False)
            
            if not input_value:
                return jsonify({"error": "URL or email is required"}), 400
            
            # Check if input is an email
            is_email = '@' in input_value
            email = None
            if is_email:
                # Extract domain from email
                email = input_value
                domain = extract_domain_from_email(email)
                if not domain:
                    return jsonify({"error": "Invalid email format"}), 400
                logger.info(f"Extracted domain '{domain}' from email '{email}'")
            else:
                # Process as domain/URL
                from urllib.parse import urlparse
                
                # Format URL properly
                if not input_value.startswith('http'):
                    input_value = 'https://' + input_value
                    
                # Extract domain
                parsed_url = urlparse(input_value)
                domain = parsed_url.netloc
                if not domain:
                    domain = parsed_url.path
                    
                # Remove www. if present
                if domain.startswith('www.'):
                    domain = domain[4:]
            
            if not domain:
                return jsonify({"error": "Invalid URL or email"}), 400
                
            url = f"https://{domain}"
            logger.info(f"Processing classification request for {domain}")
            
            # Check for domain override before any other processing
            domain_override = check_domain_override(domain)
            if domain_override:
                # Add email to response if input was an email
                if email:
                    domain_override["email"] = email
                    
                # Add website URL for clickable link
                domain_override["website_url"] = url
                
                # Return the override directly
                logger.info(f"Sending override response to client: {domain_override}")
                return jsonify(domain_override), 200
            
            # Check for existing classification if not forcing reclassification
            if not force_reclassify:
                existing_record = snowflake_conn.check_existing_classification(domain)
                
                if existing_record:
                    logger.info(f"Found existing classification for {domain}")
                    
                    # Process and return the cached result
                    from domain_classifier.storage.cache_processor import process_cached_result
                    result = process_cached_result(existing_record, domain, email, url)
                    
                    # Ensure result consistency
                    result = validate_result_consistency(result, domain)
                    
                    # Log the response for debugging
                    logger.info(f"Sending cached response to client")
                    
                    return jsonify(result), 200
            
            # Try to get content (either from DB or by crawling)
            content = None
            
            # If reclassifying or using existing content, try to get existing content first
            if force_reclassify or use_existing_content:
                try:
                    content = snowflake_conn.get_domain_content(domain)
                    if content:
                        logger.info(f"Using existing content for {domain}")
                except Exception as e:
                    logger.warning(f"Could not get existing content, will crawl instead: {e}")
                    content = None

            # If we specifically requested to use existing content but none was found
            if use_existing_content and not content:
                error_result = {
                    "domain": domain,
                    "error": "No existing content found",
                    "predicted_class": "Unknown",
                    "confidence_score": 0,
                    "confidence_scores": {
                        "Managed Service Provider": 0,
                        "Integrator - Commercial A/V": 0,
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    },
                    "explanation": f"We could not find previously stored content for {domain}. Please try recrawling instead.",
                    "low_confidence": True,
                    "no_existing_content": True,
                    "website_url": url
                }
                
                # Add email to response if input was an email
                if email:
                    error_result["email"] = email
                    
                return jsonify(error_result), 404
            
            # If no content yet and we're not using existing content, crawl the website
            error_type = None
            error_detail = None
            
            if not content and not use_existing_content:
                logger.info(f"Crawling website for {domain}")
                content, (error_type, error_detail) = crawl_website(url)
                
                if not content:
                    error_result = create_error_result(domain, error_type, error_detail, email)
                    error_result["website_url"] = url
                    return jsonify(error_result), 503  # Service Unavailable
            
            # Classify the content
            if not llm_classifier:
                error_result = {
                    "domain": domain, 
                    "error": "LLM classifier is not available",
                    "predicted_class": "Unknown",
                    "confidence_score": 0,
                    "confidence_scores": {
                        "Managed Service Provider": 0,
                        "Integrator - Commercial A/V": 0,
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    },
                    "explanation": "Our classification system is temporarily unavailable. Please try again later.",
                    "low_confidence": True,
                    "website_url": url
                }
                
                # Add email to error response if input was an email
                if email:
                    error_result["email"] = email
                    
                return jsonify(error_result), 500
                
            logger.info(f"Classifying content for {domain}")
            classification = llm_classifier.classify(content, domain)
            
            if not classification:
                error_result = {
                    "domain": domain,
                    "error": "Classification failed",
                    "predicted_class": "Unknown",
                    "confidence_score": 0,
                    "confidence_scores": {
                        "Managed Service Provider": 0,
                        "Integrator - Commercial A/V": 0,
                        "Integrator - Residential A/V": 0,
                        "Internal IT Department": 0
                    },
                    "explanation": f"We encountered an issue while analyzing {domain}.",
                    "low_confidence": True,
                    "website_url": url
                }
                
                # Add email to error response if input was an email
                if email:
                    error_result["email"] = email
                    
                return jsonify(error_result), 500
            
            # Save to Snowflake (always save, even for reclassifications)
            save_to_snowflake(domain, url, content, classification, snowflake_conn)
            
            # Process the fresh classification result
            from domain_classifier.storage.result_processor import process_fresh_result
            result = process_fresh_result(classification, domain, email, url)

            # Ensure result consistency
            result = validate_result_consistency(result, domain)
            
            # Add company description if not already present
            if "company_description" not in result:
                from domain_classifier.utils.text_processing import extract_company_description
                result["company_description"] = extract_company_description(content, result.get("explanation", ""), domain)
            
            # Log the response for debugging
            logger.info(f"Sending fresh response to client")
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Error processing request: {e}\n{traceback.format_exc()}")
            # Try to identify the error type if possible
            error_type, error_detail = detect_error_type(str(e))
            error_result = create_error_result(domain if 'domain' in locals() else "unknown", error_type, error_detail, email if 'email' in locals() else None)
            error_result["error"] = str(e)  # Add the actual error message
            if 'url' in locals():
                error_result["website_url"] = url
            return jsonify(error_result), 500
    
    @app.route('/classify-email', methods=['POST', 'OPTIONS'])
    def classify_email():
        """Alias for classify-domain that redirects email classification requests"""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            data = request.json
            email = data.get('email', '').strip()
            
            if not email:
                return jsonify({"error": "Email is required"}), 400
                
            # Create a new request with the email as URL
            new_data = {
                'url': email,
                'force_reclassify': data.get('force_reclassify', False),
                'use_existing_content': data.get('use_existing_content', False)
            }
            
            # Forward to classify_domain by calling it directly with the new data
            # This is slightly different from the original approach to avoid monkey patching
            from flask import request as flask_request
            
            # Store the original json
            original_json = flask_request.json
            
            # Create a class to simulate the request object
            class RequestProxy:
                @property
                def json(self):
                    return new_data
                    
                @property
                def method(self):
                    return 'POST'
            
            # Replace the request object temporarily
            temp_request = flask_request
            request = RequestProxy()
            
            try:
                # Call classify_domain with our proxy request
                result = classify_domain()
                return result
            finally:
                # Restore the original request
                request = temp_request
                
        except Exception as e:
            logger.error(f"Error processing email classification request: {e}\n{traceback.format_exc()}")
            error_type, error_detail = detect_error_type(str(e))
            error_result = create_error_result("unknown", error_type, error_detail)
            error_result["error"] = str(e)
            return jsonify(error_result), 500

    @app.route('/classify-and-enrich', methods=['POST', 'OPTIONS'])
    def classify_and_enrich():
        """Classify a domain and enrich it with Apollo data"""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            data = request.json
            input_value = data.get('url', '').strip()
            
            if not input_value:
                return jsonify({"error": "URL or email is required"}), 400
            
            # First perform standard classification
            # Store original request
            original_request = request.json.copy()
            
            # Call classify_domain directly
            classification_response = classify_domain()
            
            # Extract response data and status code
            if isinstance(classification_response, tuple):
                classification_result = classification_response[0].json
                status_code = classification_response[1]
            else:
                classification_result = classification_response.json
                status_code = 200
            
            # Only proceed with enrichment for successful classifications
            if status_code >= 400:
                logger.warning(f"Classification failed with status {status_code}, skipping enrichment")
                return classification_response
            
            # Extract domain from classification result
            domain = classification_result.get('domain')
            
            if not domain:
                logger.error("No domain found in classification result")
                return jsonify({"error": "Failed to extract domain from classification result"}), 500
            
            # Import Apollo connector here to avoid circular imports
            from domain_classifier.enrichment.apollo_connector import ApolloConnector
            
            # Enrich with Apollo data
            apollo = ApolloConnector()
            enrichment_data = apollo.enrich_company(domain)
            
            # Import recommendation engine
            from domain_classifier.enrichment.recommendation_engine import DomotzRecommendationEngine
            
            # Generate recommendations based on classification and Apollo data
            recommendation_engine = DomotzRecommendationEngine()
            recommendations = recommendation_engine.generate_recommendations(
                classification_result.get('predicted_class'),
                enrichment_data
            )
            
            # Add enrichment data to classification result
            classification_result['apollo_data'] = enrichment_data or {}
            classification_result['domotz_recommendations'] = recommendations
            
            # Return the enriched result
            logger.info(f"Successfully enriched and generated recommendations for {domain}")
            return jsonify(classification_result), 200
            
        except Exception as e:
            logger.error(f"Error in classify-and-enrich: {e}\n{traceback.format_exc()}")
            # Try to identify the error type
            error_type, error_detail = detect_error_type(str(e))
            # Create an error response
            error_result = create_error_result(domain if 'domain' in locals() else "unknown", 
                                             error_type, error_detail)
            error_result["error"] = str(e)  # Add the actual error message
            return jsonify(error_result), 500
            
    return app
