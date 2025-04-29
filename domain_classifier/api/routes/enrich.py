"""Enrichment related routes for the domain classifier API."""
import logging
import traceback
from flask import request, jsonify, current_app

# Import utilities
from domain_classifier.utils.error_handling import detect_error_type, create_error_result
from domain_classifier.storage.operations import save_to_snowflake
from domain_classifier.utils.final_classification import determine_final_classification

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
            
            # First perform standard classification by making an internal request
            # We'll use the routes directly from the app, rather than importing functions
            
            # Get the classify_domain function from app's view functions
            from domain_classifier.api.routes.classify import register_classify_routes
            
            # Call the registered classify-domain route directly using the Flask test client
            with app.test_client() as client:
                response = client.post('/classify-domain', json=data)
                classification_result = response.get_json()
                status_code = response.status_code
            
            # Only proceed with enrichment for successful classifications
            if status_code >= 400:
                logger.warning(f"Classification failed with status {status_code}, skipping enrichment")
                
                # Ensure final_classification is set even for error results
                if "final_classification" not in classification_result:
                    classification_result["final_classification"] = determine_final_classification(classification_result)
                    
                return jsonify(classification_result), status_code
            
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
                
            # Update final_classification based on Apollo data
            # For parked domains, check if we need to update from 1-PARKED DOMAIN w/o Apollo to 2-PARKED DOMAIN w Apollo
            if classification_result.get('final_classification') == "1-PARKED DOMAIN w/o Apollo" and company_data:
                classification_result['final_classification'] = "2-PARKED DOMAIN w Apollo"
                logger.info(f"Updated final classification to 2-PARKED DOMAIN w Apollo for {domain}")
            elif "final_classification" not in classification_result:
                # Ensure final_classification is set
                classification_result["final_classification"] = determine_final_classification(classification_result)
                logger.info(f"Added final classification: {classification_result['final_classification']} for {domain}")
                
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
            
            # Ensure final_classification is set for error results
            if "final_classification" not in error_result:
                error_result["final_classification"] = determine_final_classification(error_result)
                
            return jsonify(error_result), 500
            
    return app
