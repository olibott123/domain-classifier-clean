"""Description enhancer module for company descriptions."""
import requests
import logging
import os
import json
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def enhance_company_description(basic_description: str, apollo_data: Dict[str, Any], classification: Dict[str, Any]) -> str:
    """
    Create an enhanced company description using Apollo data and classification.
    
    Args:
        basic_description: The original basic description
        apollo_data: Company data from Apollo
        classification: Classification result
        
    Returns:
        str: Enhanced company description
    """
    enhanced_description = basic_description
    
    # Add company size and founding info if available
    if apollo_data and apollo_data.get('employee_count'):
        employee_count = apollo_data.get('employee_count')
        founded_year = apollo_data.get('founded_year', '')
        founded_phrase = f"Founded in {founded_year}, " if founded_year else ""
        
        size_description = ""
        if employee_count < 10:
            size_description = "a small"
        elif employee_count < 50:
            size_description = "a mid-sized"
        else:
            size_description = "a larger"
            
        industry = apollo_data.get('industry', '').lower()
        industry_phrase = f" in the {industry} sector" if industry else ""
        
        company_name = apollo_data.get('name', '')
        domain_name = classification.get('domain', '')
        name_to_use = company_name or domain_name
        
        # Add size and founding info to the beginning
        enhanced_description = f"{founded_phrase}{name_to_use} is {size_description} {classification.get('predicted_class', '').lower()}{industry_phrase}. {enhanced_description}"
    
    # Add technology information if available
    if apollo_data and apollo_data.get('technologies') and len(apollo_data.get('technologies')) > 0:
        tech_list = apollo_data.get('technologies')[:3]  # Take up to 3 technologies
        if tech_list:
            tech_phrase = f" They use technologies like {', '.join(tech_list)}."
            enhanced_description += tech_phrase
            
    return enhanced_description

def generate_detailed_description(classification: Dict[str, Any], 
                                apollo_data: Optional[Dict[str, Any]] = None, 
                                apollo_person_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Use Claude to generate a detailed company description.
    
    Args:
        classification: The classification result
        apollo_data: Optional company data from Apollo
        apollo_person_data: Optional person data from Apollo (kept for backwards compatibility)
        
    Returns:
        str: Detailed company description
    """
    try:
        # Build prompt with available information
        company_name = apollo_data.get('name') if apollo_data else classification.get('domain', '')
        company_type = classification.get('predicted_class', '')
        domain = classification.get('domain', '')
        
        prompt = f"""Based on the following information, write a factual, service-focused company description for {company_name}:

Business Type: {company_type}
Domain: {domain}
"""

        # Add Apollo data if available
        if apollo_data:
            industry = apollo_data.get('industry', 'Unknown')
            founded = apollo_data.get('founded_year', 'Unknown')
            size = apollo_data.get('employee_count', 'Unknown')
            technologies = apollo_data.get('technologies', [])
            
            prompt += f"""
Industry: {industry}
Founded: {founded}
Size: Approximately {size} employees
Technologies: {', '.join(technologies) if technologies else 'Unknown'}
"""

        # Add the original description
        prompt += f"""
Original Description: {classification.get('company_description', '')}

Write a factual, objective description that focuses specifically on what services or products the company provides. 
Avoid marketing language, subjective quality statements, and unnecessary adjectives. 
Focus on concrete services, technologies, or industries they serve.
Be specific about what they do rather than how well they do it.
Write approximately 75-100 words.
"""

        # Call Claude
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": os.environ.get("ANTHROPIC_API_KEY"),
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-haiku-20240307",
                "system": "You are a business analyst who writes factual, objective company descriptions focused on what services a company provides. Avoid marketing language, subjective quality statements, and unnecessary adjectives. Be specific about what they do rather than how well they do it.",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            description = result['content'][0]['text'].strip()
            logger.info(f"Successfully generated detailed description for {domain}")
            return description
        else:
            logger.error(f"Error calling Claude for detailed description: {response.status_code} - {response.text[:200]}")
            return classification.get('company_description', '')
            
    except Exception as e:
        logger.error(f"Error generating detailed description: {e}")
        return classification.get('company_description', '')
