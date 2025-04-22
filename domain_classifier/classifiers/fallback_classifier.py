import logging
import re
from typing import Dict, Any

# Set up logging
logger = logging.getLogger(__name__)

# Define indicators for different company types
# These are used for keyword-based classification
MSP_INDICATORS = [
    "managed service", "it service", "it support", "it consulting", "tech support",
    "technical support", "network", "server", "cloud", "infrastructure", "monitoring",
    "helpdesk", "help desk", "cyber", "security", "backup", "disaster recovery",
    "microsoft", "azure", "aws", "office 365", "support plan", "managed it",
    "remote monitoring", "rmm", "psa", "msp", "technology partner", "it outsourcing",
    "it provider", "email security", "endpoint protection", "business continuity",
    "ticketing", "it management", "patch management", "24/7 support", "proactive",
    "unifi", "ubiquiti", "networking", "uisp", "omada", "network management", 
    "cloud deployment", "cloud management", "network infrastructure",
    "wifi management", "wifi deployment", "network controller",
    "hosting", "hostifi", "managed hosting", "cloud hosting"
]

COMMERCIAL_AV_INDICATORS = [
    "commercial integration", "av integration", "audio visual", "audiovisual",
    "conference room", "meeting room", "digital signage", "video wall",
    "commercial audio", "commercial display", "projection system", "projector",
    "commercial automation", "room scheduling", "presentation system", "boardroom",
    "professional audio", "business audio", "commercial installation", "enterprise",
    "huddle room", "training room", "av design", "control system", "av consultant",
    "crestron", "extron", "biamp", "amx", "polycom", "cisco", "zoom room",
    "teams room", "corporate", "business communication", "commercial sound"
]

RESIDENTIAL_AV_INDICATORS = [
    "home automation", "smart home", "home theater", "residential integration",
    "home audio", "home sound", "custom installation", "home control", "home cinema",
    "residential av", "whole home audio", "distributed audio", "multi-room",
    "lighting control", "home network", "home wifi", "entertainment system",
    "sonos", "control4", "savant", "lutron", "residential automation", "smart tv",
    "home entertainment", "consumer", "residential installation", "home integration"
]

# New indicators for service business detection
SERVICE_BUSINESS_INDICATORS = [
    "service", "provider", "solutions", "consulting", "management", "support",
    "agency", "professional service", "firm", "consultancy", "outsourced"
]

# Indicators for internal IT potential
INTERNAL_IT_POTENTIAL_INDICATORS = [
    "enterprise", "corporation", "corporate", "company", "business", "organization",
    "staff", "team", "employees", "personnel", "department", "division",
    "global", "nationwide", "locations", "offices", "headquarters"
]

# Indicators that explicitly should NOT lead to specific classifications
NEGATIVE_INDICATORS = {
    "vacation rental": "NOT_RESIDENTIAL_AV",
    "holiday rental": "NOT_RESIDENTIAL_AV",
    "hotel booking": "NOT_RESIDENTIAL_AV",
    "holiday home": "NOT_RESIDENTIAL_AV",
    "vacation home": "NOT_RESIDENTIAL_AV",
    "travel agency": "NOT_RESIDENTIAL_AV",
    "booking site": "NOT_RESIDENTIAL_AV",
    "book your stay": "NOT_RESIDENTIAL_AV",
    "accommodation": "NOT_RESIDENTIAL_AV",
    "reserve your": "NOT_RESIDENTIAL_AV",
    "feriebolig": "NOT_RESIDENTIAL_AV",  # Danish for vacation home
    "ferie": "NOT_RESIDENTIAL_AV",       # Danish for vacation
    
    "add to cart": "NOT_MSP",
    "add to basket": "NOT_MSP",
    "shopping cart": "NOT_MSP",
    "free shipping": "NOT_MSP",
    "checkout": "NOT_MSP",
    
    "our products": "NOT_SERVICE",
    "product catalog": "NOT_SERVICE",
    "manufacturer": "NOT_SERVICE"
}

def parse_free_text(text: str, domain: str = None) -> Dict[str, Any]:
    """
    Parse classification from free-form text response when JSON parsing fails.
    
    Args:
        text: The text to parse
        domain: Optional domain name for context
        
    Returns:
        dict: The parsed classification
    """
    text_lower = text.lower()
    
    # Check for parked domain indicators
    if any(phrase in text_lower for phrase in ["parked domain", "under construction", "domain for sale"]):
        from domain_classifier.classifiers.decision_tree import create_parked_domain_result
        return create_parked_domain_result(domain)
        
    # Check for process failure indicators
    if any(phrase in text_lower for phrase in ["process did not complete", "couldn't process", "failed to process"]):
        from domain_classifier.classifiers.decision_tree import create_process_did_not_complete_result
        return create_process_did_not_complete_result(domain)
    
    # First determine if it's a service business
    is_service = True  # Default assumption
    non_service_indicators = [
        "not a service business", 
        "non-service business", 
        "not a managed service provider",
        "not an integrator",
        "doesn't provide services", 
        "doesn't offer services",
        "transportation", "logistics", "shipping", "trucking",
        "vacation rental", "travel agency", "hotel", "accommodation"
    ]
    
    for indicator in non_service_indicators:
        if indicator in text_lower:
            is_service = False
            logger.info(f"Text indicates non-service business: '{indicator}'")
            break
            
    # Extract predicted class
    if is_service:
        # Look for service business type
        class_patterns = [
            (r"managed service provider|msp", "Managed Service Provider"),
            (r"commercial a\/?v|commercial integrator", "Integrator - Commercial A/V"),
            (r"residential a\/?v|residential integrator", "Integrator - Residential A/V")
        ]
        
        predicted_class = None
        for pattern, class_name in class_patterns:
            if re.search(pattern, text_lower):
                predicted_class = class_name
                logger.info(f"Found predicted class in text: {class_name}")
                break
                
        # Default if no clear match
        if not predicted_class:
            predicted_class = "Managed Service Provider"  # Most common fallback
            logger.warning(f"No clear service type found, defaulting to MSP")
    else:
        predicted_class = "Internal IT Department"
        
    # Extract or estimate internal IT potential for non-service businesses
    internal_it_potential = None
    if not is_service:
        # Look for explicit mention
        it_potential_match = re.search(r"internal\s+it\s+potential.*?(\d+)", text_lower)
        if it_potential_match:
            internal_it_potential = int(it_potential_match.group(1))
            logger.info(f"Found internal IT potential score: {internal_it_potential}")
        else:
            # Estimate based on business descriptions
            enterprise_indicators = ["enterprise", "corporation", "company", "business", "organization", "large"]
            it_indicators = ["technology", "digital", "online", "systems", "platform"]
            
            enterprise_count = sum(1 for term in enterprise_indicators if term in text_lower)
            it_count = sum(1 for term in it_indicators if term in text_lower)
            
            if enterprise_count > 0 or it_count > 0:
                # Scale 0-10 to 20-80
                base_score = min(10, enterprise_count + it_count)
                internal_it_potential = 20 + (base_score * 6)
            else:
                internal_it_potential = 30  # Default low-medium
                
            logger.info(f"Estimated internal IT potential: {internal_it_potential}")
    else:
        # For service businesses, internal IT potential is always 0
        internal_it_potential = 0
            
    # Calculate confidence scores
    confidence_scores = {}
    
    if is_service:
        # Count keyword matches in the text
        msp_score = sum(1 for keyword in MSP_INDICATORS if keyword in text_lower)
        commercial_score = sum(1 for keyword in COMMERCIAL_AV_INDICATORS if keyword in text_lower)
        residential_score = sum(1 for keyword in RESIDENTIAL_AV_INDICATORS if keyword in text_lower)
        
        total_score = max(1, msp_score + commercial_score + residential_score)
        
        # Calculate proportional scores
        msp_conf = 0.30 + (0.5 * msp_score / total_score) if msp_score > 0 else 0.08
        comm_conf = 0.30 + (0.5 * commercial_score / total_score) if commercial_score > 0 else 0.08
        resi_conf = 0.30 + (0.5 * residential_score / total_score) if residential_score > 0 else 0.08
        
        confidence_scores = {
            "Managed Service Provider": int(msp_conf * 100),
            "Integrator - Commercial A/V": int(comm_conf * 100),
            "Integrator - Residential A/V": int(resi_conf * 100),
            "Internal IT Department": 0  # Service businesses always have 0 for Internal IT Department
        }
        
        # Ensure predicted class has highest confidence
        if predicted_class in confidence_scores:
            highest_score = max(confidence_scores.items(), key=lambda x: x[1] if x[0] != "Internal IT Department" else 0)[0]
            confidence_scores[predicted_class] = max(confidence_scores[predicted_class], confidence_scores[highest_score] + 5)
    else:
        # Low confidence scores for all service categories
        confidence_scores = {
            "Managed Service Provider": 5,
            "Integrator - Commercial A/V": 3,
            "Integrator - Residential A/V": 2,
            "Internal IT Department": internal_it_potential or 30
        }
        
    # Extract or generate explanation
    from domain_classifier.utils.text_processing import extract_explanation, generate_explanation
    explanation = extract_explanation(text)
    if not explanation or len(explanation) < 100:
        explanation = generate_explanation(predicted_class, domain, is_service, internal_it_potential)
        
    # Generate company description
    from domain_classifier.utils.text_processing import extract_company_description
    company_description = extract_company_description(text, explanation, domain)
        
    # Calculate max confidence
    if is_service and predicted_class in confidence_scores:
        max_confidence = confidence_scores[predicted_class] / 100.0
    else:
        max_confidence = 0.5  # Medium confidence for non-service
        
    return {
        "processing_status": 2,  # Success
        "is_service_business": is_service,
        "predicted_class": predicted_class,
        "internal_it_potential": internal_it_potential,
        "confidence_scores": confidence_scores,
        "llm_explanation": explanation,
        "company_description": company_description,
        "detection_method": "text_parsing",
        "low_confidence": is_service and max_confidence < 0.4,
        "max_confidence": max_confidence
    }

def create_non_service_result(domain: str, text_content: str) -> Dict[str, Any]:
    """
    Create a standardized result for non-service businesses.
    
    Args:
        domain: The domain name
        text_content: The text content to analyze
        
    Returns:
        dict: Standardized non-service business result
    """
    # Calculate internal IT potential
    text_lower = text_content.lower()
    enterprise_terms = ["company", "business", "corporation", "organization", "enterprise"]
    size_terms = ["global", "nationwide", "locations", "staff", "team", "employees", "department", "division"]
    tech_terms = ["technology", "digital", "platform", "system", "software", "online"]
    
    enterprise_count = sum(1 for term in enterprise_terms if term in text_lower)
    size_count = sum(1 for term in size_terms if term in text_lower)
    tech_count = sum(1 for term in tech_terms if term in text_lower)
    
    # Scale up to 1-100 range
    internal_it_potential = 20 + min(60, (enterprise_count * 5) + (size_count * 3) + (tech_count * 4))
    
    # Higher IT potential for transportation companies
    if "transport" in text_lower or "logistics" in text_lower or "trucking" in text_lower or "shipping" in text_lower:
        internal_it_potential = max(internal_it_potential, 55)  # Transport companies usually need IT
        logger.info(f"Adjusted internal IT potential for transportation company: {internal_it_potential}")
    
    # Higher potential for financial services
    if "bank" in text_lower or "finance" in text_lower or "financial" in text_lower or "insurance" in text_lower:
        internal_it_potential = max(internal_it_potential, 70)  # Financial companies need significant IT
        logger.info(f"Adjusted internal IT potential for financial company: {internal_it_potential}")
        
    # Lower potential for very small businesses
    if "small" in text_lower and "business" in text_lower:
        internal_it_potential = min(internal_it_potential, 40)
        logger.info(f"Adjusted internal IT potential for small business: {internal_it_potential}")
        
    # Generate an appropriate explanation
    if "transport" in text_lower or "logistics" in text_lower or "trucking" in text_lower:
        explanation = f"{domain or 'This company'} appears to be a transportation and logistics service provider, not a service/management business or an A/V integrator. The company focuses on physical transportation, shipping, or logistics rather than providing IT or A/V services. This type of company typically has moderate to significant internal IT needs to manage their operations, logistics systems, and fleet management."
        company_description = f"{domain or 'This company'} is a transportation and logistics provider offering shipping and freight services."
    elif "vacation" in text_lower or "hotel" in text_lower or "accommodation" in text_lower or "booking" in text_lower:
        explanation = f"{domain or 'This company'} appears to be in the travel, tourism, or hospitality industry, not an IT service provider or A/V integrator. The company focuses on providing accommodations, vacation rentals, or travel-related services rather than IT or A/V services. This type of business typically has low to moderate internal IT needs to maintain booking systems and websites."
        company_description = f"{domain or 'This company'} is a travel or hospitality business offering accommodations or vacation-related services."
    elif "retail" in text_lower or "shop" in text_lower or "store" in text_lower or "product" in text_lower:
        explanation = f"{domain or 'This company'} appears to be a retail or e-commerce business, not an IT service provider or A/V integrator. The company sells products rather than providing IT or A/V services. This type of business typically has varying levels of internal IT needs depending on their size and online presence."
        company_description = f"{domain or 'This company'} is a retail or e-commerce business selling products to consumers."
    else:
        explanation = f"{domain or 'This company'} does not appear to be a service business in the IT or A/V integration space. There is no evidence that it provides managed services, IT support, or audio/visual integration to clients. Rather, it appears to be a company that might have its own internal IT needs (estimated potential: {internal_it_potential}/100)."
        company_description = f"{domain or 'This company'} appears to be a business entity that doesn't provide IT or A/V services to clients."
        
    # Create Internal IT Department result with minimal confidence scores for service categories
    return {
        "processing_status": 2,  # Success
        "is_service_business": False,
        "predicted_class": "Internal IT Department",
        "internal_it_potential": internal_it_potential,
        "confidence_scores": {
            "Managed Service Provider": 5,
            "Integrator - Commercial A/V": 3,
            "Integrator - Residential A/V": 2,
            "Internal IT Department": internal_it_potential
        },
        "llm_explanation": explanation,
        "company_description": company_description,
        "detection_method": "non_service_detection",
        "low_confidence": True,
        "max_confidence": 0.8  # Reasonably confident in non-service classification
    }

def fallback_classification(text_content: str, domain: str = None) -> Dict[str, Any]:
    """
    Fallback classification method when LLM classification fails.
    
    Args:
        text_content: The text content to classify
        domain: Optional domain name for context
        
    Returns:
        dict: The classification results
    """
    logger.info("Using fallback classification method")
    
    # First determine if it's likely a service business
    text_lower = text_content.lower()
    
    # Count service-related terms
    service_count = sum(1 for term in SERVICE_BUSINESS_INDICATORS if term in text_lower)
    
    # Is this likely a service business?
    is_service = service_count >= 2
    
    if domain:
        domain_lower = domain.lower()
        # Domain name hints for service business
        if any(term in domain_lower for term in ["service", "tech", "it", "consult", "support", "solutions"]):
            is_service = True
            logger.info(f"Domain name indicates service business: {domain}")
            
        # Special case for travel/vacation domains
        vacation_terms = ["vacation", "holiday", "rental", "booking", "hotel", "travel"]
        if any(term in domain_lower for term in vacation_terms):
            is_service = False
            logger.info(f"Domain name indicates vacation/travel business (non-service): {domain}")
            
        # Special case for transportation/logistics
        transport_terms = ["trucking", "transport", "logistics", "shipping", "freight", "delivery"]
        if any(term in domain_lower for term in transport_terms):
            is_service = False
            logger.info(f"Domain name indicates transportation/logistics business (non-service): {domain}")
    
    logger.info(f"Fallback classification service business determination: {is_service}")
    
    if is_service:
        # Start with default confidence scores
        confidence = {
            "Managed Service Provider": 0.35,
            "Integrator - Commercial A/V": 0.25,
            "Integrator - Residential A/V": 0.15
        }
        
        # Count keyword occurrences
        msp_count = sum(1 for keyword in MSP_INDICATORS if keyword in text_lower)
        commercial_count = sum(1 for keyword in COMMERCIAL_AV_INDICATORS if keyword in text_lower)
        residential_count = sum(1 for keyword in RESIDENTIAL_AV_INDICATORS if keyword in text_lower)
        
        total_count = max(1, msp_count + commercial_count + residential_count)
        
        # Check for negative indicators
        for indicator, neg_class in NEGATIVE_INDICATORS.items():
            if indicator in text_lower:
                logger.info(f"Found negative indicator: {indicator} -> {neg_class}")
                # Apply rule based on negative indicator
                if neg_class == "NOT_RESIDENTIAL_AV":
                    # Drastically reduce Residential AV score if vacation rental indicators are found
                    confidence["Integrator - Residential A/V"] = 0.05
                    residential_count = 0  # Reset for score calculation below
                elif neg_class == "NOT_MSP":
                    # Reduce MSP score for e-commerce indicators
                    confidence["Managed Service Provider"] = 0.05
                    msp_count = 0
                elif neg_class == "NOT_SERVICE":
                    # If strong indicator of non-service, override the classification
                    logger.info(f"Found strong non-service indicator: {indicator}")
                    return create_non_service_result(domain, text_content)
        
        # Domain name analysis
        domain_hints = {"msp": 0, "commercial": 0, "residential": 0}
        
        if domain:
            domain_lower = domain.lower()
            
            # MSP related domain terms
            if any(term in domain_lower for term in ["it", "tech", "computer", "service", "cloud", "cyber", "network", "support", "wifi", "unifi", "hosting", "host", "fi", "net"]):
                domain_hints["msp"] += 3
                
            # Commercial A/V related domain terms
            if any(term in domain_lower for term in ["av", "audio", "visual", "video", "comm", "business", "enterprise", "corp"]):
                domain_hints["commercial"] += 2
                
            # Residential A/V related domain terms - be careful with vacation terms
            if any(term in domain_lower for term in ["home", "residential", "smart", "theater", "cinema"]):
                # Don't boost residential score for vacation rental domains
                if not any(term in domain_lower for term in ["vacation", "holiday", "rental", "booking", "hotel"]):
                    domain_hints["residential"] += 2
                
        # Adjust confidence based on keyword counts and domain hints
        confidence["Managed Service Provider"] = 0.25 + (0.35 * msp_count / total_count) + (0.1 * domain_hints["msp"])
        confidence["Integrator - Commercial A/V"] = 0.15 + (0.35 * commercial_count / total_count) + (0.1 * domain_hints["commercial"])
        confidence["Integrator - Residential A/V"] = 0.10 + (0.35 * residential_count / total_count) + (0.1 * domain_hints["residential"])
            
        # Special case handling
        if domain:
            if "hostifi" in domain.lower():
                confidence["Managed Service Provider"] = 0.85
                confidence["Integrator - Commercial A/V"] = 0.08
                confidence["Integrator - Residential A/V"] = 0.05
            
            # Special handling for vacation rental domains    
            elif any(term in domain.lower() for term in ["vacation", "holiday", "rental", "booking", "hotel"]):
                # Ensure not classified as Residential A/V
                confidence["Integrator - Residential A/V"] = 0.05
                
            # Special handling for transport/logistics domains
            elif any(term in domain.lower() for term in ["trucking", "transport", "logistics"]):
                # This should not be a service business at all
                return create_non_service_result(domain, text_content)
                
        # Additional content checks for non-service businesses
        if "transport" in text_lower or "logistics" in text_lower or "shipping" in text_lower or "trucking" in text_lower:
            # This is likely a transportation/logistics company, not a service provider
            logger.info("Content suggests transportation/logistics business, reclassifying as non-service")
            return create_non_service_result(domain, text_content)
            
        # Determine predicted class based on highest confidence
        predicted_class = max(confidence.items(), key=lambda x: x[1])[0]
        
        # Apply the adjustment logic to ensure meaningful differences between categories
        if predicted_class == "Managed Service Provider" and confidence["Managed Service Provider"] > 0.5:
            confidence["Integrator - Residential A/V"] = min(confidence["Integrator - Residential A/V"], 0.12)
        
        # Generate explanation
        from domain_classifier.utils.text_processing import generate_explanation
        explanation = generate_explanation(predicted_class, domain, is_service)
        explanation += " (Note: This classification is based on our fallback system, as detailed analysis was unavailable.)"
        
        # Generate company description
        from domain_classifier.utils.text_processing import extract_keywords_company_description
        company_description = extract_keywords_company_description(text_content, predicted_class, domain)
        
        # Convert decimal confidence to 1-100 range
        confidence_scores = {k: int(v * 100) for k, v in confidence.items()}
        
        # Add Internal IT Department with score 0 for service businesses
        confidence_scores["Internal IT Department"] = 0
        
        return {
            "processing_status": 2,  # Success
            "is_service_business": True,
            "predicted_class": predicted_class,
            "internal_it_potential": 0,  # Set to 0 for service businesses
            "confidence_scores": confidence_scores,
            "llm_explanation": explanation,
            "company_description": company_description,
            "detection_method": "fallback",
            "low_confidence": True,
            "max_confidence": confidence_scores[predicted_class] / 100.0
        }
    else:
        return create_non_service_result(domain, text_content)
