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
    # Early validation - if content is None or empty, return default structure
    if not content:
        logger.warning(f"No content available for AI extraction for domain: {domain}")
        return {
            "name": domain.split('.')[0].capitalize(),
            "source": "fallback_extraction"
        }
    
    # First try pattern matching for common data points
    company_data = _extract_with_patterns(content, domain)
    
    # Always try AI extraction for completeness
    ai_data = _extract_with_ai(content, domain, classification, company_data)
    
    # Merge AI data, prioritizing existing pattern-matched data
    for key, value in ai_data.items():
        if key != "source":  # Skip the source field
            # Only add AI data if:
            # 1. The field is not already populated, or
            # 2. Current value is None and new value has content
            if (key not in company_data or company_data[key] is None) and value:
                # Extra validation before adding
                if _is_valid_field_value(key, value):
                    company_data[key] = value
    
    # Final data quality check
    company_data = _final_quality_check(company_data, domain)
    
    # If we couldn't extract data successfully, at least ensure the company name is set
    if not company_data.get("name"):
        company_data["name"] = domain.split('.')[0].capitalize()
    
    # Set proper source field if it exists
    if "source" not in company_data:
        company_data["source"] = "combined_extraction"
    
    return company_data

def _final_quality_check(data: Dict[str, Any], domain: str) -> Dict[str, Any]:
    """
    Perform a final quality check on the extracted data.
    
    Args:
        data: The extracted company data
        domain: The domain name
        
    Returns:
        dict: Validated company data
    """
    # Check for invalid values in location fields
    suspicious_values = ["windows", "linux", "unix", "null", "undefined", 
                         "object", "string", "number", "boolean"]
    
    for field in ["city", "state", "country"]:
        if field in data and isinstance(data[field], str):
            value = data[field].lower()
            if value in suspicious_values:
                logger.warning(f"Removed suspicious {field} value: {value}")
                data[field] = None
    
    # Check for code-like address values
    if "address" in data and isinstance(data["address"], str):
        if re.search(r'[{}<>=;]', data["address"]):
            # Check if it contains any street indicators
            if not re.search(r'\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|lane|ln\.?|drive|dr\.?|blvd\.?)\b', 
                            data["address"], re.IGNORECASE):
                logger.warning(f"Removed code-like address value: {data['address']}")
                data["address"] = None
    
    return data

def _is_valid_field_value(key: str, value: Any) -> bool:
    """
    Validate that a field value is reasonable and not code/script content.
    
    Args:
        key: Field key
        value: Field value to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not value or not isinstance(value, str):
        return True  # Non-string values pass through
        
    # Check for suspicious code-like patterns and characters
    code_patterns = [
        r'==', # JavaScript equality comparison
        r'!=', # JavaScript inequality
        r'===', # JavaScript strict equality
        r'!==', # JavaScript strict inequality
        r'[}{]\s*\)', # Code block ending
        r';\s*\)', # JavaScript statement ending
        r'</script>', # HTML script tag
        r'</style>', # HTML style tag
        r'function\s*\(', # JavaScript function
        r'if\s*\(', # JavaScript if statement
        r'return\s', # JavaScript return statement
        r'var\s+', # JavaScript variable declaration
        r'const\s+', # JavaScript const declaration
        r'let\s+', # JavaScript let declaration
        r'https?://', # URLs (common in code)
        r'window\.', # JavaScript window object
        r'document\.', # JavaScript document object
    ]
    
    # For city, state, country - reject values like "windows" which are likely errors
    if key in ["city", "state", "country"]:
        # Reject common programming terms as location names
        invalid_locations = ["windows", "linux", "unix", "null", "undefined", "object", 
                             "string", "number", "boolean", "true", "false"]
        if value.lower() in invalid_locations:
            logger.warning(f"Rejected invalid {key} value: {value}")
            return False
    
    # Special case for address - extra validation
    if key == "address":
        # Reject if doesn't look like a physical address at all
        has_street_indicator = re.search(r'(street|st\.?|avenue|ave\.?|road|rd\.?|lane|ln\.?|drive|dr\.?|blvd\.?|boulevard|way|place|plaza|suite)', 
                                         value, re.IGNORECASE)
        has_number = re.search(r'\d+', value)
        
        # If we have code-like characters, and no address patterns, it's likely invalid
        if (re.search(r'[}{)(;=]', value) or 
            any(re.search(pattern, value) for pattern in code_patterns)):
            
            # Only pass if it has strong indicators of being a real address
            if not has_street_indicator or not has_number:
                logger.warning(f"Rejected code-like address: {value}")
                return False
    
    # General validation for all fields
    if re.search(r'[}{)(;=<>]', value):
        # Check for common code patterns
        for pattern in code_patterns:
            if re.search(pattern, value):
                logger.warning(f"Rejected suspicious value for {key}: {value}")
                return False
    
    # Value seems valid
    return True

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

    # Extract phone number - IMPROVED PATTERN
    phone_patterns = [
        r'(?:phone|tel|telephone|call)(?:\s|:|\n)+(\+?[\d\s\(\)\-\.]{10,20})',
        r'(\+?[\d\s\(\)\-\.]{10,20})(?=\s*(?:phone|tel|telephone|ext|extension))',
        r'(\+?1[\.\-\s]?(?:\(\d{3}\)|\d{3})[\.\-\s]?\d{3}[\.\-\s]?\d{4})'  # US phone format
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
    
    # Extract email - IMPROVED PATTERN
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
    
    # Extract address - IMPROVED PATTERN WITH BETTER VALIDATION
    address_patterns = [
        # Street address with number - most reliable pattern
        r'(?:address|location|headquarters|offices?|located at|find us at)(?:\s|:)+([0-9]+\s+[a-zA-Z]+\s+(?:street|st\.?|avenue|ave\.?|road|rd\.?|boulevard|blvd\.?|drive|dr\.?|lane|ln\.?|place|pl\.?|court|ct\.?)(?:\s|\.|,)[^,\n]{3,40}(?:,\s*[a-zA-Z\s]+){1,3})',
        
        # PO Box pattern
        r'(?:address|location|headquarters|offices?)(?:\s|:)+(?:P\.?O\.?\s+Box\s+[0-9]+[^,\n]{5,40}(?:,\s*[a-zA-Z\s]+){1,3})',
        
        # Simple comma-separated address
        r'(?:address|location|headquarters|offices?)(?:\s|:)+([^,;]{5,50},\s*[^,;]{5,50},\s*[^,;]{2,20})'
    ]
    
    for pattern in address_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            full_address = match.group(1).strip()
            
            # Further validation to ensure it looks like an address
            if re.search(r'\d+\s+\w+', full_address):  # Must have a number + word (likely street number + name)
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
    
    # Extract founded year - IMPROVED PATTERN
    founded_patterns = [
        r'(?:founded|established|since|est\.|started)(?:\s|:|\n)+(?:in\s+)?([0-9]{4})',
        r'(?:since|est\.|established|founded)\s+([0-9]{4})',
        r'founded\s+in\s+([0-9]{4})',
        r'established\s+in\s+([0-9]{4})'
    ]
    
    for pattern in founded_patterns:
        match = re.search(pattern, content_lower)
        if match:
            founded_year = match.group(1).strip()
            try:
                year = int(founded_year)
                if 1800 <= year <= 2025:  # Validate reasonable year range
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
        "construction": ["construction", "building", "contractor", "architecture"],
        "security": ["security", "surveillance", "alarm", "camera", "access control", "monitoring"]
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
        r'(?:welcome to|about) ([^.!?\n<>]{3,40})',
        r'([a-zA-Z0-9\s&]+)(?:\s+is\s+a\s+(?:leading|premier|trusted))',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            # Validate the extracted name
            if (
                len(name) > 3 and len(name) < 50 and 
                not re.match(r'^(home|index|welcome|about|contact)$', name.lower()) and
                not re.search(r'[<>{}();]', name)  # No code-like characters
            ):
                return name
    
    # Fallback: Use domain name with capitalization
    domain_parts = domain.split('.')
    if len(domain_parts) > 0:
        # Convert "alaskansecurity" to "Alaskan Security"
        name_parts = re.findall(r'[A-Z][a-z]*|[a-z]+', domain_parts[0])
        if name_parts:
            return ' '.join(part.capitalize() for part in name_parts)
    
    return domain.split('.')[0].capitalize()

def _is_minimal_company_data(company_data: Dict[str, Any]) -> bool:
    """Check if extracted company data is minimal and requires AI enhancement."""
    # Count how many fields have actual data
    essential_fields = ["name", "address", "phone", "email", "industry"]
    filled_essential_fields = sum(1 for field in essential_fields if company_data.get(field))
    
    # If fewer than 3 essential fields have data, consider it minimal
    return filled_essential_fields < 3

def _extract_with_ai(content: str, domain: str, classification: Dict[str, Any], existing_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract company data using Claude."""
    try:
        # Get API key from environment
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("No ANTHROPIC_API_KEY available for AI company data extraction")
            return {}
        
        # Limit content to avoid token limits
        if content and len(content) > 8000:
            content = content[:8000]
        
        company_description = classification.get("company_description", "")
        predicted_class = classification.get("predicted_class", "Unknown")
        
        # Create the prompt for Claude with extremely specific instructions about formatting
        prompt = f"""Based ONLY on the following website content for {domain}, extract structured company information.
        
Website Content: {content}

Additional Context:
- Company Description: {company_description}
- Business Type: {predicted_class}

Extract the following company information ONLY from the provided website content. Do NOT make anything up or get creative. If you cannot find the information in the content, leave it blank.

CRITICAL INSTRUCTIONS:
1. For address field, extract ONLY a physical street address like "123 Main St, City, State"
2. NEVER include HTML, JavaScript, code snippets, or symbols like ==, (), {{}}, ; etc. in ANY field
3. Addresses must include street numbers and names
4. City, state and country should be actual place names, not computer terms like "windows"
5. If you are uncertain about any value, use null instead

Return your response in this JSON format:
{{
  "name": "Company name",
  "address": "Full street address",
  "city": "City name only",
  "state": "State or region name only",
  "country": "Country name only",
  "postal_code": "Postal/ZIP code",
  "phone": "Phone number",
  "email": "Contact email",
  "founded_year": Year founded (numeric only),
  "employee_count": Approximate employee count (numeric only),
  "industry": "Primary industry"
}}

For any fields you can't find in the content, use null. Never make up information.
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
                "system": "You are an expert at extracting factual company information from website content. You never make up information and only extract what's explicitly stated in the provided content. You must be extremely careful to avoid extracting code, HTML elements, or JavaScript as part of company data. Never include code fragments, symbols like ==, {}, (), or JavaScript syntax in any field. For addresses, only extract physical street addresses.",
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
                    json_str = json_match.group(1)
                    
                    # Additional cleaning before parsing to handle common issues
                    # Remove any code-like artifacts from the JSON string
                    json_str = re.sub(r'==\s*[\'"][^\'"]*[\'"]', 'null', json_str)
                    json_str = re.sub(r'function\s*\(.*?\)', 'null', json_str)
                    
                    # Clean up potential script tags in values
                    json_str = re.sub(r'<script[^>]*>.*?</script>', 'null', json_str, flags=re.DOTALL)
                    
                    # Fix common JavaScript syntax that breaks JSON
                    json_str = re.sub(r':\s*undefined', ': null', json_str)
                    json_str = re.sub(r':\s*NaN', ': null', json_str)
                    
                    # Now parse the cleaned JSON
                    extracted_data = json.loads(json_str)
                    
                    # Clean the data with additional validation
                    clean_data = {}
                    for key, value in extracted_data.items():
                        if value is None or value == "null" or value == "":
                            clean_data[key] = None
                        elif isinstance(value, str) and _is_valid_field_value(key, value):
                            # Extra aggressive cleaning for all string fields
                            value = re.sub(r'==[^=]*', '', value)  # Remove JS equality expressions
                            value = re.sub(r'function\s*\(.*?\)', '', value)  # Remove JS functions
                            value = re.sub(r'[{}<>]', '', value)  # Remove brackets
                            value = re.sub(r'https?://\S+', '', value)  # Remove URLs
                            value = re.sub(r'\s+', ' ', value).strip()  # Normalize whitespace
                            
                            # Revalidate after cleaning
                            if value and _is_valid_field_value(key, value):
                                clean_data[key] = value
                        elif isinstance(value, (int, float)) and key in ['founded_year', 'employee_count']:
                            # Only allow numeric values for numeric fields
                            clean_data[key] = value
                    
                    # Add source field
                    clean_data["source"] = "ai_extraction"
                    
                    # Final validation - ensure we're not returning garbage data
                    if "city" in clean_data and clean_data["city"] == "windows":
                        clean_data["city"] = None
                    if "state" in clean_data and clean_data["state"] == "windows":
                        clean_data["state"] = None
                    if "country" in clean_data and clean_data["country"] == "windows":
                        clean_data["country"] = None
                    
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
