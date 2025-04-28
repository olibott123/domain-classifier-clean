from flask import request, jsonify
import logging
import os
import traceback

# Set up logging
logger = logging.getLogger(__name__)

# Import domain utilities
from domain_classifier.utils.domain_utils import extract_domain_from_email, extract_domain_from_url
from domain_classifier.utils.error_handling import detect_error_type, create_error_result

# Import configuration
from domain_classifier.config.overrides import check_domain_override

# Import services
from domain_classifier.classifiers.llm_classifier import LLMClassifier
from domain_classifier.storage.snowflake_connector import SnowflakeConnector
from domain_classifier.crawlers.apify_crawler import crawl_website
from domain_classifier.storage.operations import save_to_snowflake, query_similar_domains
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

# Initialize Vector DB connector
try:
    from domain_classifier.storage.vector_db import VectorDBConnector
    vector_db_conn = VectorDBConnector()
    logger.info(f"Vector DB connector initialized and connected: {getattr(vector_db_conn, 'connected', False)}")
except Exception as e:
    logger.error(f"Error initializing Vector DB connector: {e}")
    vector_db_conn = None

def register_routes(app):
    """Register all routes with the app."""
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Simple health check endpoint"""
        return jsonify({
            "status": "ok", 
            "llm_available": llm_classifier is not None,
            "snowflake_connected": getattr(snowflake_conn, 'connected', False),
            "vector_db_connected": getattr(vector_db_conn, 'connected', False)
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
            
            # Save to Snowflake and Vector DB (always save, even for reclassifications)
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
            
            # Determine if input is an email
            is_email = '@' in input_value
            email = input_value if is_email else None
            
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
            
            # Extract domain and email from classification result
            domain = classification_result.get('domain')
            email = classification_result.get('email')  # This will be set if the input was an email
            
            if not domain:
                logger.error("No domain found in classification result")
                return jsonify({"error": "Failed to extract domain from classification result"}), 500
            
            # Import Apollo connector here to avoid circular imports
            from domain_classifier.enrichment.apollo_connector import ApolloConnector
            from domain_classifier.enrichment.description_enhancer import enhance_company_description, generate_detailed_description
            
            # Initialize Apollo connector
            apollo = ApolloConnector()
            
            # Enrich with Apollo company data
            company_data = apollo.enrich_company(domain)
            
            # If we have an email, also get person data
            person_data = None
            if email:
                logger.info(f"Looking up person data for email: {email}")
                person_data = apollo.search_person(email)
            
            # Import recommendation engine
            from domain_classifier.enrichment.recommendation_engine import DomotzRecommendationEngine
            
            # Generate recommendations based on classification and Apollo data
            recommendation_engine = DomotzRecommendationEngine()
            recommendations = recommendation_engine.generate_recommendations(
                classification_result.get('predicted_class'),
                company_data
            )
            
            # Step 1: First enhance with Apollo data
            if company_data:
                logger.info(f"Enhancing description with Apollo data for {domain}")
                basic_enhanced_description = enhance_company_description(
                    classification_result.get("company_description", ""),
                    company_data,
                    classification_result
                )
                classification_result["company_description"] = basic_enhanced_description
            
            # Step 2: Then use Claude to generate a more detailed description
            try:
                detailed_description = generate_detailed_description(
                    classification_result,
                    company_data,
                    person_data
                )
                
                if detailed_description and len(detailed_description) > 50:
                    classification_result["company_description"] = detailed_description
                    logger.info(f"Updated description with detailed Claude-generated version for {domain}")
            except Exception as desc_error:
                logger.error(f"Error generating detailed description: {desc_error}")
                # Keep the basic enhanced description if the detailed one fails
            
            # Add enrichment data to classification result
            classification_result['apollo_data'] = company_data or {}
            
            # Add person data if available
            if person_data:
                classification_result['apollo_person_data'] = person_data
            
            # Add recommendations
            classification_result['domotz_recommendations'] = recommendations
            
            # Save the enriched classification to Snowflake
            from domain_classifier.storage.operations import save_to_snowflake
            url = f"https://{domain}"
            content = snowflake_conn.get_domain_content(domain)
            
            # Save the enhanced data to Snowflake (with Apollo data)
            save_to_snowflake(
                domain=domain, 
                url=url, 
                content=content, 
                classification=classification_result,
                snowflake_conn=snowflake_conn,
                apollo_company_data=company_data,
                apollo_person_data=person_data
            )
            
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
    
    @app.route('/query-similar-domains', methods=['POST', 'OPTIONS'])
    def find_similar_domains():
        """Find domains similar to the given query text or domain"""
        # Handle preflight requests
        if request.method == 'OPTIONS':
            return '', 204
        
        try:
            data = request.json
            
            # Get query parameters
            query_text = data.get('query_text', '').strip()
            domain = data.get('domain', '').strip()
            top_k = data.get('top_k', 5)
            filter_criteria = data.get('filter', None)
            
            # Validate input
            if not query_text and not domain:
                return jsonify({"error": "Either query_text or domain must be provided"}), 400
                
            # If domain is provided but not query_text, get domain content for the query
            if domain and not query_text:
                # Get domain content from Snowflake
                try:
                    content = snowflake_conn.get_domain_content(domain)
                    if content:
                        query_text = content
                    else:
                        return jsonify({
                            "error": "No content found for domain",
                            "domain": domain
                        }), 404
                except Exception as e:
                    logger.error(f"Error getting domain content for similarity query: {e}")
                    return jsonify({
                        "error": f"Failed to retrieve content for domain: {str(e)}",
                        "domain": domain
                    }), 500
            
            # Query for similar domains
            results = query_similar_domains(
                query_text=query_text,
                top_k=top_k,
                filter=filter_criteria
            )
            
            # Return the results
            return jsonify({
                "query": query_text[:100] + "..." if len(query_text) > 100 else query_text,
                "domain": domain if domain else None,
                "top_k": top_k,
                "results": results,
                "result_count": len(results)
            }), 200
            
        except Exception as e:
            logger.error(f"Error querying similar domains: {e}\n{traceback.format_exc()}")
            return jsonify({
                "error": str(e),
                "results": []
            }), 500
            
    return app
