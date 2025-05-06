"""AI-based company data extractor from website content."""
import re
import logging
import os
import json
import requests
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

def extract_company_data_from_content(content: str, domain: str, classification: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract company data from website content using pattern matching and AI.
    
    Args:
        content: The website content
        domain: The domain name
        classification: The classification result with explanation
        
    Returns:
        dict: Company data extracted from content
    """
    # First try pattern matching for common data points
    company_data = _extract_with_patterns(content, domain)
    
    # If pattern matching yielded minimal results, try AI extraction
    if _is_minimal_company_data(company_data):
        ai_data = _extract_with_ai(content, domain, classification, company_data)
        # Merge AI data, prioritizing existing pattern-matched data
        for key, value in ai_data.items():
            if not company_data.get(key) and value:
                company_data[key] = value
    
    return company_data

def _extract_with_patterns(content: str, domain: str) -> Dict[str, Any]:
    """Extract company data using regex pattern matching."""
    company_data = {
        "name": None,
        "address": None,
        "city": None,
        "state": None,
        "country": None,
        "postal_code": None,
        "phone": None,
        "email": None,
        "founded_year": None,
        "employee_count": None,
        "industry": None,
        "source": "extracted_from_website"
    }
    
    # Lowercase content for case-insensitive matching
    content_lower = content.lower()
    
    # Extract company name
    company_name = _extract_company_name(content, domain)
    if company_name:
        company_data["name"] = company_name

    # Extract phone number
    phone_patterns = [
        r'(?:phone|tel|telephone|call)(?:\s|:|\n)+(\+?[\d\s\(\)\-\.]{10,20})',
        r'(\+?[\d\s\(\)\-\.]{10,20})(?=\s*(?:phone|tel|telephone|ext|extension))',
        r'(?<![a-zA-Z])(\+?[\d\s\(\)\-\.]{10,20})(?![a-zA-Z])'
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, content_lower)
        if match:
            phone = match.group(1).strip()
            # Clean the phone number
            phone = re.sub(r'[^\d\+\(\)\-\.\s]', '', phone)
            if len(re.sub(r'[^\d]', '', phone)) >= 7:  # Ensure it has enough digits
                company_data["phone"] = phone
                break
    
    # Extract email
    email_pattern = r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}'
    email_matches = re.findall(email_pattern, content)
    
    # Filter out emails from the same domain (likely service emails)
    company_emails = [email for email in email_matches if domain.lower() in email.lower()]
    
    if company_emails:
        # Prioritize info@ or contact@ emails
        for email in company_emails:
            if email.lower().startswith(('info@', 'contact@', 'hello@')):
                company_data["email"] = email
                break
        
        # If no priority email was found, use the first one
        if not company_data["email"] and company_emails:
            company_data["email"] = company_emails[0]
    
    # Extract address
    address_patterns = [
        r'(?:address|location|headquarters)(?:\s|:)+([^,\n]+,[^,\n]+,[^,\n]+(?:,[^,\n]+)?)',
        r'(?:address|location|headquarters)(?:\s|:)+([^,\n]+,[^,\n]+(?:,[^,\n]+)?)',
        r'([0-9]+\s+[a-zA-Z]+\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|place|pl|court|ct)\.?(?:,|\s+)[a-zA-Z\s]+(?:,|\s+)[a-zA-Z]{2}(?:,|\s+)[0-9]{5}(?:-[0-9]{4})?)',
        r'([0-9]+\s+[a-zA-Z]+\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|place|pl|court|ct)\.?(?:,|\s+)[a-zA-Z\s]+(?:,|\s+)[a-zA-Z]{2})',
        r'([0-9]+\s+[a-zA-Z]+\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|lane|ln|place|pl|court|ct))'
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            full_address = match.group(1).strip()
            
            # Store the full address
            company_data["address"] = full_address
            
            # Try to extract city, state, country from the address
            address_parts = [part.strip() for part in full_address.split(',')]
            if len(address_parts) >= 3:
                # Last part might contain postal code and country
                last_part = address_parts[-1].strip()
                if re.search(r'[0-9]', last_part):  # Contains numbers, likely postal code
                    company_data["postal_code"] = last_part
                    if len(address_parts) >= 4:
                        company_data["country"] = address_parts[-2].strip()
                else:
                    company_data["country"] = last_part
                
                # Second-to-last part is likely city or state
                if len(address_parts) >= 2:
                    state_part = address_parts[-2].strip()
                    # If it's 2 letters, it's likely a US state code
                    if len(state_part) <= 3 and state_part.isalpha():
                        company_data["state"] = state_part
                        if len(address_parts) >= 3:
                            company_data["city"] = address_parts[-3].strip()
                    else:
                        company_data["city"] = state_part
            
            break
    
    # Extract founded year
    founded_patterns = [
        r'(?:founded|established|since|est\.)(?:\s|:|\n)+(?:in\s+)?([0-9]{4})',
        r'(?:since|est\.|established|founded)\s+([0-9]{4})'
    ]
    
    for pattern in founded_patterns:
        match = re.search(pattern, content_lower)
        if match:
            founded_year = match.group(1).strip()
            try:
                year = int(founded_year)
                if 1800 <= year <= 2023:  # Validate reasonable year range
                    company_data["founded_year"] = year
                    break
            except ValueError:
                continue
    
    # Extract industry
    industry_keywords = {
        "technology": ["technology", "software", "saas", "tech company", "it services", "information technology"],
        "healthcare": ["healthcare", "medical", "health", "hospital", "clinic", "wellness"],
        "finance": ["finance", "banking", "investment", "financial", "insurance", "wealth management"],
        "education": ["education", "school", "university", "college", "learning", "training"],
        "manufacturing": ["manufacturing", "factory", "production", "industrial"],
        "retail": ["retail", "shop", "store", "e-commerce", "ecommerce"],
        "hospitality": ["hospitality", "hotel", "restaurant", "catering", "food service"],
        "real estate": ["real estate", "property", "realty", "housing"],
        "consulting": ["consulting", "consultant", "advisory", "professional services"],
        "entertainment": ["entertainment", "media", "music", "film", "movie"],
        "transportation": ["transportation", "logistics", "shipping", "freight"],
        "construction": ["construction", "building", "contractor", "architecture"]
    }
    
    industry_counts = {industry: 0 for industry in industry_keywords}
    
    for industry, keywords in industry_keywords.items():
        for keyword in keywords:
            count = content_lower.count(keyword)
            industry_counts[industry] += count
    
    # Determine the most likely industry
    max_industry = max(industry_counts.items(), key=lambda x: x[1])
    if max_industry[1] > 2:  # Threshold to avoid false positives
        company_data["industry"] = max_industry[0]
    
    return company_data

def _extract_company_name(content: str, domain: str) -> Optional[str]:
    """Extract company name from content."""
    # Try to extract from common patterns
    name_patterns = [
        r'<title>(.*?)([-|]|</title>)',
        r'(?:welcome to|about) ([^.!?\n<>]+)',
        r'([a-zA-Z0-9\s]+)(?:\s+is\s+a\s+(?:leading|premier|trusted))',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Validate the extracted name
            if (
                len(name) > 3 and len(name) < 50 and 
                not re.match(r'^(home|index|welcome|about|contact)$', name.lower())
            ):
                return name
    
    # Fallback: Use domain name with capitalization
    domain_name = domain.split('.')[0].capitalize()
    return domain_name

def _is_minimal_company_data(company_data: Dict[str, Any]) -> bool:
    """Check if extracted company data is minimal and requires AI enhancement."""
    # Count how many fields have actual data
    filled_fields = sum(1 for value in company_data.values() if value)
    
    # If fewer than 4 fields have data, consider it minimal
    return filled_fields < 4

def _extract_with_ai(content: str, domain: str, classification: Dict[str, Any], existing_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract company data using Claude."""
    try:
        # Get API key from environment
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("No ANTHROPIC_API_KEY available for AI company data extraction")
            return {}
        
        # Limit content to avoid token limits
        if content and len(content) > 12000:
            content = content[:12000]
        
        company_description = classification.get("company_description", "")
        predicted_class = classification.get("predicted_class", "Unknown")
        
        # Create the prompt for Claude
        prompt = f"""Based ONLY on the following website content for {domain}, extract structured company information.
        
Website Content: {content}

Additional Context:
- Company Description: {company_description}
- Business Type: {predicted_class}

Extract the following company information ONLY from the provided website content. Do NOT make anything up or get creative. If you cannot find the information in the content, leave it blank.

Return your response in this JSON format:
{{
  "name": "Company name",
  "address": "Full street address",
  "city": "City",
  "state": "State or region",
  "country": "Country",
  "postal_code": "Postal/ZIP code",
  "phone": "Phone number",
  "email": "Contact email",
  "founded_year": Year founded (numeric),
  "employee_count": Approximate employee count (numeric),
  "industry": "Primary industry"
}}

For any fields you can't find in the content, use null. Do not make up information.
"""
        
        # Call Claude API
        logger.info(f"Calling Claude API to extract company data for {domain}")
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            },
            json={
                "model": "claude-3-haiku-20240307",
                "system": "You are an expert at extracting factual company information from website content. You never make up information and only extract what's explicitly stated in the provided content.",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1000,
                "temperature": 0.1
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            response_text = result['content'][0]['text'].strip()
            
            # Extract JSON from response
            import re
            json_match = re.search(r'({[\s\S]*})', response_text)
            if json_match:
                try:
                    extracted_data = json.loads(json_match.group(1))
                    
                    # Clean the data
                    clean_data = {}
                    for key, value in extracted_data.items():
                        if value is None or value == "null" or value == "":
                            clean_data[key] = None
                        else:
                            clean_data[key] = value
                    
                    # Add source field
                    clean_data["source"] = "ai_extraction"
                    
                    logger.info(f"Successfully extracted AI company data for {domain}")
                    return clean_data
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from Claude response for {domain}")
            
        else:
            logger.error(f"Error from Claude API: {response.status_code}")
            
        # Return empty dict if we couldn't get data
        return {}
        
    except Exception as e:
        logger.error(f"Error extracting company data with AI: {e}")
        return {}
