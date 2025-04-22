"""Text processing utilities for domain classification."""
import re
import logging
import json
from typing import Optional, Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON string by removing control characters and fixing common issues.
    
    Args:
        json_str: The JSON string to clean
        
    Returns:
        str: The cleaned JSON string
    """
    # Replace control characters
    cleaned = re.sub(r'[\x00-\x1F\x7F]', '', json_str)
    
    # Replace single quotes with double quotes
    cleaned = re.sub(r"'([^']*)':", r'"\1":', cleaned)
    
    # Fix trailing commas
    cleaned = re.sub(r',\s*}', '}', cleaned)
    cleaned = re.sub(r',\s*]', ']', cleaned)
    
    # Fix missing quotes around property names
    cleaned = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', cleaned)
    
    # Replace unescaped newlines in strings
    cleaned = re.sub(r'(".*?)\n(.*?")', r'\1\\n\2', cleaned, flags=re.DOTALL)
    
    # Handle decimal values without leading zero
    cleaned = re.sub(r':\s*\.(\d+)', r': 0.\1', cleaned)
    
    # Try to fix quotes within quotes in explanation fields by escaping them
    if '"llm_explanation"' in cleaned:
        # Complex regex to find the explanation field and properly escape quotes
        explanation_pattern = r'"llm_explanation"\s*:\s*"(.*?)"(?=,|\s*})'
        match = re.search(explanation_pattern, cleaned, re.DOTALL)
        if match:
            explanation_text = match.group(1)
            # Escape any unescaped quotes within the explanation
            fixed_explanation = explanation_text.replace('"', '\\"')
            # Replace back in the original string
            cleaned = cleaned.replace(explanation_text, fixed_explanation)
    
    return cleaned

def extract_json(text: str) -> Optional[str]:
    """
    Extract JSON from text response.
    
    Args:
        text: The text to extract JSON from
        
    Returns:
        str: The extracted JSON string, or None if not found
    """
    # Try multiple patterns to extract JSON
    json_patterns = [
        r'({[\s\S]*"predicted_class"[\s\S]*})',  # Most general pattern
        r'```(?:json)?\s*({[\s\S]*})\s*```',     # For markdown code blocks
        r'({[\s\S]*"confidence_scores"[\s\S]*})' # Alternative key pattern
    ]
    
    for pattern in json_patterns:
        json_match = re.search(pattern, text, re.DOTALL)
        if json_match:
            return json_match.group(1)
            
    return None

def detect_minimal_content(content: str) -> bool:
    """
    Detect if domain has minimal content.
    
    Args:
        content: The website content
        
    Returns:
        bool: True if the domain has minimal content
    """
    if not content or len(content.strip()) < 100:
        logger.info(f"Content is very short: {len(content) if content else 0} characters")
        return True
        
    # Count words in content
    words = re.findall(r'\b\w+\b', content.lower())
    unique_words = set(words)
    
    # Return true if few words or unique words
    if len(words) < 50:
        logger.info(f"Content has few words ({len(words)}), likely minimal content")
        return True
        
    if len(unique_words) < 30:
        logger.info(f"Content has few unique words ({len(unique_words)}), likely minimal content")
        return True
            
    return False

def extract_company_description(content: str, explanation: str, domain: str) -> str:
    """
    Extract or generate a concise company description.
    
    Args:
        content: The website content
        explanation: The classification explanation
        domain: The domain name
        
    Returns:
        str: A concise company description
    """
    # First try to extract from LLM explanation
    description_patterns = [
        r'company description: (.*?)(?=\n|\.|$)',
        r'(?:the company|this company|the business|this business) (provides|offers|specializes in|focuses on|is) ([^.]+)',
        r'(?:appears to be|seems to be) (a|an) ([^.]+)'
    ]
    
    for pattern in description_patterns:
        match = re.search(pattern, explanation, re.IGNORECASE)
        if match and len(match.group(0)) > 20:
            # Clean up the description
            desc = match.group(0).strip()
            # Convert to third person if needed
            desc = re.sub(r'^we ', f"{domain} ", desc, flags=re.IGNORECASE)
            desc = re.sub(r'^our ', f"{domain}'s ", desc, flags=re.IGNORECASE)
            
            # Ensure it starts with the domain name
            if not desc.lower().startswith(domain.lower()):
                desc = f"{domain} {desc}"
                
            return desc
    
    # If explanation doesn't yield a good description, try to extract from website content
    if content:
        # Look for an "about us" paragraph
        about_patterns = [
            r'about\s+us[^.]*(?:[^.]*\.){1,2}',
            r'who\s+we\s+are[^.]*(?:[^.]*\.){1,2}',
            r'our\s+company[^.]*(?:[^.]*\.){1,2}'
        ]
        
        for pattern in about_patterns:
            match = re.search(pattern, content, re.IGNORECASE)
            if match and len(match.group(0)) > 30:
                # Clean up the description
                desc = match.group(0).strip()
                # Convert to third person if needed
                desc = re.sub(r'^we ', f"{domain} ", desc, flags=re.IGNORECASE)
                desc = re.sub(r'^our ', f"{domain}'s ", desc, flags=re.IGNORECASE)
                
                # Ensure it starts with the domain name
                if not desc.lower().startswith(domain.lower()):
                    desc = f"{domain} {desc}"
                    
                # Limit length
                if len(desc) > 200:
                    desc = desc[:197] + "..."
                    
                return desc
    
    # Fall back to using information from the explanation
    predicted_class = ""
    if "managed service provider" in explanation.lower():
        predicted_class = "a Managed Service Provider offering IT services and solutions"
    elif "commercial a/v" in explanation.lower():
        predicted_class = "a Commercial A/V Integrator providing audiovisual solutions for businesses"
    elif "residential a/v" in explanation.lower():
        predicted_class = "a Residential A/V Integrator specializing in home automation and entertainment systems"
    elif "internal it" in explanation.lower():
        predicted_class = "a business with internal IT needs rather than an IT service provider"
    else:
        predicted_class = "a business whose specific activities couldn't be determined with high confidence"
    
    return f"{domain} appears to be {predicted_class}."

def extract_keywords_company_description(content: str, predicted_class: str, domain: str) -> str:
    """
    Generate a company description based on keywords in the content.
    
    Args:
        content: The website content
        predicted_class: The predicted class
        domain: The domain name
        
    Returns:
        str: A keyword-based company description
    """
    content_lower = content.lower()
    
    # Identify industry keywords
    industry_keywords = []
    industry_patterns = [
        (r'(healthcare|medical|health\s+care|patient)', "healthcare"),
        (r'(education|school|university|college|learning|teaching)', "education"),
        (r'(finance|banking|investment|financial|insurance)', "finance"),
        (r'(retail|ecommerce|e-commerce|online\s+store|shop)', "retail"),
        (r'(manufacturing|factory|production|industrial)', "manufacturing"),
        (r'(government|public\s+sector|federal|state|municipal)', "government"),
        (r'(hospitality|hotel|restaurant|tourism)', "hospitality"),
        (r'(technology|software|saas|cloud|application)', "technology"),
        (r'(construction|building|architecture|engineering)', "construction"),
        (r'(transportation|logistics|shipping|freight)', "transportation")
    ]
    
    for pattern, keyword in industry_patterns:
        if re.search(pattern, content_lower):
            industry_keywords.append(keyword)
    
    # Create appropriate description based on predicted class and keywords
    if predicted_class == "Managed Service Provider":
        services = []
        if "network" in content_lower or "networking" in content_lower:
            services.append("network management")
        if "security" in content_lower or "cyber" in content_lower:
            services.append("cybersecurity")
        if "cloud" in content_lower:
            services.append("cloud services")
        if "support" in content_lower or "helpdesk" in content_lower:
            services.append("technical support")
        
        service_text = ""
        if services:
            service_text = f" specializing in {', '.join(services)}"
            
        industry_text = ""
        if industry_keywords:
            industry_text = f" for the {', '.join(industry_keywords)} {'industry' if len(industry_keywords) == 1 else 'industries'}"
            
        return f"{domain} is a Managed Service Provider{service_text}{industry_text}."
        
    elif predicted_class == "Integrator - Commercial A/V":
        solutions = []
        if "conference" in content_lower or "meeting" in content_lower:
            solutions.append("conference room systems")
        if "digital signage" in content_lower or "display" in content_lower:
            solutions.append("digital signage")
        if "video" in content_lower and "wall" in content_lower:
            solutions.append("video walls")
        if "automation" in content_lower:
            solutions.append("automation systems")
            
        solution_text = ""
        if solutions:
            solution_text = f" providing {', '.join(solutions)}"
            
        industry_text = ""
        if industry_keywords:
            industry_text = f" for the {', '.join(industry_keywords)} {'sector' if len(industry_keywords) == 1 else 'sectors'}"
            
        return f"{domain} is a Commercial A/V Integrator{solution_text}{industry_text}."
        
    elif predicted_class == "Integrator - Residential A/V":
        solutions = []
        if "home theater" in content_lower or "cinema" in content_lower:
            solutions.append("home theaters")
        if "automation" in content_lower or "smart home" in content_lower:
            solutions.append("smart home automation")
        if "audio" in content_lower:
            solutions.append("audio systems")
        if "lighting" in content_lower:
            solutions.append("lighting control")
            
        solution_text = ""
        if solutions:
            solution_text = f" specializing in {', '.join(solutions)}"
            
        return f"{domain} is a Residential A/V Integrator{solution_text} for home environments."
        
    else:  # Internal IT Department / non-service business
        business_type = ""
        if industry_keywords:
            business_type = f"a {', '.join(industry_keywords)} business"
        else:
            business_type = "a business entity"
            
        return f"{domain} appears to be {business_type} that doesn't provide IT or A/V services to clients."
