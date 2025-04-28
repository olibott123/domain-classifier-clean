import logging
import traceback
from flask import request, jsonify

# Import utilities
from domain_classifier.utils.error_handling import detect_error_type, create_error_result
from domain_classifier.storage.operations import save_to_snowflake

# Import the classify route function
from domain_classifier.api.routes.classify import classify_domain as classify_domain_route

# Set up logging
logger = logging.getLogger(__name__)

def register_enrich_routes(app, snowflake_conn):
    """Register enrichment related routes."""
    
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
            
            # Call classify_domain route function directly
            classification_response = classify_domain_route()
            
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
            
            # Extract domain, email and crawler type from classification result
            domain = classification_result.get('domain')
            email = classification_result.get('email')  # This will be set if the input was an email
            crawler_type = classification_result.get('crawler_type')  # Get crawler type from classification result
            
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
            
            # Don't look up person data to save Apollo credits
            person_data = None
            
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
                    None  # No person data passed
                )
                
                if detailed_description and len(detailed_description) > 50:
                    classification_result["company_description"] = detailed_description
                    logger.info(f"Updated description with detailed Claude-generated version for {domain}")
            except Exception as desc_error:
                logger.error(f"Error generating detailed description: {desc_error}")
                # Keep the basic enhanced description if the detailed one fails
            
            # Add enrichment data to classification result
            classification_result['apollo_data'] = company_data or {}
            
            # Add recommendations
            classification_result['domotz_recommendations'] = recommendations
            
            # Make sure the crawler_type is preserved from the original classification
            if not classification_result.get('crawler_type') and crawler_type:
                classification_result['crawler_type'] = crawler_type
                
            # Save the enriched classification to Snowflake
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
                crawler_type=crawler_type,  # Explicitly pass the crawler_type from the original classification
                classifier_type="claude-llm-enriched"
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
            
    return app
