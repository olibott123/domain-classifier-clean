import requests
import logging
import json
import time
import re
from typing import Dict, Any, List, Optional
import os
import csv
import socket
from urllib.parse import urlparse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMClassifier:
    def __init__(self, api_key: str = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the LLM classifier with API key and model.
        
        Args:
            api_key: The API key for Claude API
            model: The Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No API key provided. LLM classification will not be available.")
            
        self.model = model
        
        # Define indicators for different company types
        # These are used as fallbacks and domain name analysis
        self.msp_indicators = [
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
        
        self.commercial_av_indicators = [
            "commercial integration", "av integration", "audio visual", "audiovisual",
            "conference room", "meeting room", "digital signage", "video wall",
            "commercial audio", "commercial display", "projection system", "projector",
            "commercial automation", "room scheduling", "presentation system", "boardroom",
            "professional audio", "business audio", "commercial installation", "enterprise",
            "huddle room", "training room", "av design", "control system", "av consultant",
            "crestron", "extron", "biamp", "amx", "polycom", "cisco", "zoom room",
            "teams room", "corporate", "business communication", "commercial sound"
        ]
        
        self.residential_av_indicators = [
            "home automation", "smart home", "home theater", "residential integration",
            "home audio", "home sound", "custom installation", "home control", "home cinema",
            "residential av", "whole home audio", "distributed audio", "multi-room",
            "lighting control", "home network", "home wifi", "entertainment system",
            "sonos", "control4", "savant", "lutron", "residential automation", "smart tv",
            "home entertainment", "consumer", "residential installation", "home integration"
        ]
        
        # New indicators for service business detection
        self.service_business_indicators = [
            "service", "provider", "solutions", "consulting", "management", "support",
            "agency", "professional service", "firm", "consultancy", "outsourced"
        ]
        
        # Indicators for internal IT potential
        self.internal_it_potential_indicators = [
            "enterprise", "corporation", "corporate", "company", "business", "organization",
            "staff", "team", "employees", "personnel", "department", "division",
            "global", "nationwide", "locations", "offices", "headquarters"
        ]
        
        # Indicators that explicitly should NOT lead to specific classifications
        self.negative_indicators = {
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

    def _load_examples_from_knowledge_base(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Load examples from knowledge base CSV file with enhanced logging.
        
        Returns:
            Dict mapping categories to lists of example domains with content
        """
        examples = {
            "Managed Service Provider": [],
            "Integrator - Commercial A/V": [],
            "Integrator - Residential A/V": [],
            "Non-Service Business": []  # New category for non-service examples
        }
        
        try:
            kb_path = "knowledge_base.csv"
            logger.info(f"Attempting to load knowledge base from {kb_path}")
            
            # Fall back to example_domains.csv if knowledge_base.csv doesn't exist
            if not os.path.exists(kb_path):
                kb_path = "example_domains.csv"
                logger.warning(f"Knowledge base not found, falling back to {kb_path}")
                
                # If using example_domains (which may not have content), create synthetic content
                if os.path.exists(kb_path):
                    with open(kb_path, 'r', encoding='utf-8') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            company_type = row.get('company_type', '')
                            if company_type in examples:
                                examples[company_type].append({
                                    'domain': row.get('domain', ''),
                                    'content': f"This is a {company_type} specializing in solutions for their clients."
                                })
                    logger.info(f"Loaded {sum(len(examples[cat]) for cat in examples)} examples from {kb_path}")
                    for category in examples:
                        logger.info(f"Loaded {len(examples[category])} examples for category {category}")
                    return examples
            
            # Regular case - load from knowledge_base.csv which has real content
            if os.path.exists(kb_path):
                with open(kb_path, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    headers = next(reader)  # Skip header row
                    
                    for row in reader:
                        if len(row) >= 3:  # domain, company_type, content
                            company_type = row[1]
                            if company_type in examples:
                                examples[company_type].append({
                                    'domain': row[0],
                                    'content': row[2]
                                })
                
                logger.info(f"Successfully loaded {sum(len(examples[cat]) for cat in examples)} examples from {kb_path}")
                for category in examples:
                    logger.info(f"Loaded {len(examples[category])} examples for category {category}")
            else:
                logger.warning(f"Neither knowledge_base.csv nor example_domains.csv found")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
        
        # Ensure we have at least some examples for each category
        for category in examples:
            if not examples[category]:
                # Fallback synthetic examples if no real ones exist
                if category == "Managed Service Provider":
                    examples[category].append({
                        'domain': 'example-msp.com',
                        'content': 'We provide managed IT services including network management, cybersecurity, cloud solutions, and 24/7 technical support for businesses of all sizes. Our team of certified engineers can handle all your technology needs from help desk support to cloud infrastructure management.'
                    })
                elif category == "Integrator - Commercial A/V":
                    examples[category].append({
                        'domain': 'example-commercial-av.com',
                        'content': 'We design and implement professional audio-visual solutions for businesses, including conference rooms, digital signage systems, and corporate presentation technologies. Our commercial clients rely on us for integrated communication solutions across their enterprises.'
                    })
                elif category == "Integrator - Residential A/V":
                    examples[category].append({
                        'domain': 'example-residential-av.com',
                        'content': 'We specialize in smart home automation and high-end home theater installations for residential clients, including lighting control, whole-home audio, and custom home cinema rooms. Our team creates extraordinary entertainment experiences in luxury homes.'
                    })
                else:  # Non-Service Business
                    examples[category].append({
                        'domain': 'example-nonservice.com',
                        'content': 'We are an online retailer selling consumer electronics and accessories. Browse our wide selection of products for your home and office needs. Shop now for great deals on computers, phones, and entertainment gadgets with fast shipping to your door.'
                    })
        
        return examples

    def classify(self, text_content: str, domain: str = None) -> Dict[str, Any]:
        """
        Classify text content following the new decision tree approach.
        
        Args:
            text_content: The text content to classify
            domain: Optional domain name for context
            
        Returns:
            dict: The classification results including predicted class and confidence scores
        """
        logger.info(f"Starting classification for domain: {domain or 'unknown'}")
        
        # STEP 1: Check if processing can complete
        if not text_content:
            logger.warning(f"No content provided for domain: {domain or 'unknown'}")
            return self._create_process_did_not_complete_result(domain)
            
        # Cache lowercase text for repeated use
        self.text_lower = text_content.lower()
        
        # STEP 2: Check if this is a parked/minimal domain
        if self._is_parked_domain(text_content):
            logger.info(f"Domain {domain or 'unknown'} is detected as a parked domain")
            return self._create_parked_domain_result(domain)
            
        is_minimal_content = self.detect_minimal_content(text_content)
        if is_minimal_content:
            logger.info(f"Domain {domain or 'unknown'} has minimal content")
            
        # Special case handling for specific domains
        if domain:
            domain_result = self._check_special_domain_cases(domain, text_content)
            if domain_result:
                return domain_result
            
        # STEP 3: Determine if it's a service/management business
        # Let's use the LLM for this rather than just keywords
        try:
            if not self.api_key:
                raise ValueError("No API key provided")
                
            # Load examples from knowledge base
            examples = self._load_examples_from_knowledge_base()
            
            # Log knowledge base usage
            total_examples = sum(len(examples[cat]) for cat in examples)
            logger.info(f"Loaded {total_examples} examples from knowledge base")
            for category in examples:
                logger.info(f"  - {category}: {len(examples[category])} examples")
            
            # Build system prompt with the new decision tree approach
            system_prompt = self._build_decision_tree_prompt(examples)
                
            # Limit the text content to avoid token limits
            max_chars = 12000
            if len(text_content) > max_chars:
                text_content = text_content[:max_chars]
                
            # Create the request to the Claude API
            url = "https://api.anthropic.com/v1/messages"
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": self.model,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": f"Domain name: {domain or 'unknown'}\n\nWebsite content to classify: {text_content}"}
                ],
                "max_tokens": 1500,
                "temperature": 0.1
            }
            
            # Make the request to Claude
            logger.info(f"Making request to Claude API for domain {domain or 'unknown'}")
            response = requests.post(url, headers=headers, json=data, timeout=60)
            response_data = response.json()
            
            if "error" in response_data:
                logger.error(f"Error from Claude API: {response_data['error']}")
                raise Exception(f"Claude API error: {response_data['error']}")
                
            # Extract the response text
            if "content" in response_data and len(response_data["content"]) > 0:
                text_response = response_data["content"][0]["text"]
                logger.info(f"Received response from Claude API for {domain}")
            else:
                logger.error("No content in Claude response")
                raise Exception("No content in Claude response")
                
            # Try to extract JSON from the response
            json_str = self._extract_json(text_response)
            
            if json_str:
                try:
                    # Try to parse the JSON
                    parsed_json = json.loads(self.clean_json_string(json_str))
                    
                    # Validate and normalize the parsed JSON
                    parsed_json = self._validate_classification(parsed_json, domain)
                    
                    # Add detection method
                    parsed_json["detection_method"] = "llm_classification"
                    
                    # Set low_confidence flag based on highest score
                    max_confidence = parsed_json.get("max_confidence", 0.5)
                    if isinstance(max_confidence, str):
                        try:
                            max_confidence = float(max_confidence)
                        except (ValueError, TypeError):
                            max_confidence = 0
                            
                    parsed_json["low_confidence"] = max_confidence < 0.4 if parsed_json.get("is_service_business", False) else True
                    
                    logger.info(f"Successful LLM classification for {domain or 'unknown'}: {parsed_json['predicted_class']}")
                    
                    # Check if this is a service business
                    is_service = parsed_json.get("is_service_business", True)
                    logger.info(f"Domain {domain} service business check: {is_service}")
                    
                    # Ensure the explanation has the step-by-step format
                    parsed_json = self._ensure_step_format(parsed_json, domain)
                    
                    # Return the validated classification
                    return parsed_json
                    
                except Exception as e:
                    logger.error(f"Error parsing LLM response: {e}")
                    logger.error(f"JSON string: {json_str}")
            
            # If we get here, JSON parsing failed, try free text parsing
            logger.warning("Could not find JSON in LLM response, falling back to text parsing")
            parsed_result = self._parse_free_text(text_response, domain)
            if is_minimal_content:
                parsed_result["detection_method"] = "text_parsing_with_minimal_content"
            else:
                parsed_result["detection_method"] = "text_parsing"
            
            # Ensure the explanation has the step-by-step format
            parsed_result = self._ensure_step_format(parsed_result, domain)
            
            logger.info(f"Extracted classification from free text: {parsed_result['predicted_class']}")
            return parsed_result
            
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
            # Fall back to keyword-based classification
            result = self._fallback_classification(text_content, domain)
            if is_minimal_content:
                result["detection_method"] = result["detection_method"] + "_with_minimal_content"
                
            # Ensure the explanation has the step-by-step format
            result = self._ensure_step_format(result, domain)
            
            return result
    
    def _build_decision_tree_prompt(self, examples: Dict[str, List[Dict[str, str]]]) -> str:
        """
        Build a prompt that implements the decision tree approach as explained by the boss.
        
        Args:
            examples: Knowledge base examples
            
        Returns:
            str: The system prompt for the LLM
        """
        # Implement the new decision tree prompt
        system_prompt = """You are an expert business analyst who specializes in categorizing companies based on website content.
You need to follow a specific decision tree to classify the business:

STEP 1: Determine if processing can complete
- If there is insufficient content to analyze, output "Process Did Not Complete"

STEP 2: Check if this is a parked/inactive domain
- If the domain is parked, under construction, or for sale, classify as "Parked Domain"

STEP 3: Determine if the company is a SERVICE/MANAGEMENT business:
- Service businesses provide ongoing services, support, or managed solutions to clients
- They typically focus on outsourced functions, consulting, or delivering specialized expertise
- Examples: IT service providers, consultants, agencies, outsourced service providers

STEP 4: If it IS a service business, classify into ONE of these specific service categories:
1. Managed Service Provider (MSP): IT services, support, network management, cybersecurity, cloud services, etc.
2. Integrator - Commercial A/V: Audio/visual solutions for businesses, conference rooms, commercial automation, etc.
3. Integrator - Residential A/V: Home theater, smart home, residential automation, whole-home audio, etc.

STEP 5: If it is NOT a service business, determine if it's a business that could have an internal IT department:
- Medium to large businesses with multiple employees and locations typically can have internal IT
- Enterprises, corporations, manufacturers, retailers, healthcare providers, financial institutions, etc.

IMPORTANT GUIDELINES:
- Service businesses should have a Corporate IT score of 0 (as they are providing services, not consuming IT internally)
- Non-service businesses should have a Corporate IT score between 1-100 indicating their internal IT potential
- Vacation rental services, travel agencies, and hospitality businesses are NOT A/V Integrators
- Web design agencies and general marketing firms are NOT typically MSPs unless they explicitly offer IT services
- Media production companies are NOT necessarily A/V integrators
- Purely online retailers or e-commerce sites generally don't provide services and are NOT MSPs
- Transportation and logistics companies are NOT service providers in the IT sense and should be classified as Non-Service Business

Here are examples of correctly classified companies:"""

        # Add examples to the system prompt
        for category in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V", "Non-Service Business"]:
            if category in examples and examples[category]:
                system_prompt += f"\n\n## {category} examples:\n"
                for example in examples[category][:2]:  # Just use 2 examples per category
                    snippet = example.get('content', '')[:300].replace('\n', ' ').strip()
                    system_prompt += f"\nDomain: {example.get('domain', 'example.com')}\nContent snippet: {snippet}...\nClassification: {category}\n"
        
        # Add instructions for output format
        system_prompt += """\n\nYou MUST provide your analysis in JSON format with the following structure:
{
  "processing_status": [0 if processing failed, 1 if domain is parked, 2 if classification successful],
  "is_service_business": [true/false - whether this is a service/management business],
  "predicted_class": [if is_service_business=true: "Managed Service Provider", "Integrator - Commercial A/V", or "Integrator - Residential A/V" | if is_service_business=false: "Non-Service Business" | if processing_status=0: "Process Did Not Complete" | if processing_status=1: "Parked Domain"],
  "internal_it_potential": [if is_service_business=false: score from 1-100 indicating likelihood the business could have internal IT department | if is_service_business=true: 0],
  "confidence_scores": {
    "Integrator - Commercial A/V": [Integer from 1-100, only relevant if is_service_business=true],
    "Integrator - Residential A/V": [Integer from 1-100, only relevant if is_service_business=true],
    "Managed Service Provider": [Integer from 1-100, only relevant if is_service_business=true],
    "Corporate IT": [0 if is_service_business=true, integer from 1-100 if is_service_business=false]
  },
  "llm_explanation": "A detailed explanation of your reasoning and classification process"
}

IMPORTANT INSTRUCTIONS:
1. You MUST follow the decision tree exactly as specified.
2. For service businesses, provide DIFFERENT confidence scores for each category.
3. For service businesses, the Corporate IT score MUST be 0.
4. For non-service businesses, all service-type confidence scores should be very low (1-10).
5. Internal IT potential should be 0 for service businesses and scored 1-100 for non-service businesses.
6. Your llm_explanation must detail your decision process through each step.
7. Mention specific evidence from the text that supports your classification.
8. If the business appears to be a transport, logistics, manufacturing, or retail company, you MUST classify it as a Non-Service Business, not as an MSP.
9. Your explanation MUST be formatted with STEP 1, STEP 2, etc. clearly labeled for each stage of the decision tree.

YOUR RESPONSE MUST BE A SINGLE VALID JSON OBJECT WITH NO OTHER TEXT BEFORE OR AFTER."""

        return system_prompt

    def _ensure_step_format(self, classification: Dict[str, Any], domain: str = None) -> Dict[str, Any]:
        """
        Ensure the explanation follows the step-by-step format.
        
        Args:
            classification: The classification dictionary
            domain: Optional domain name for context
            
        Returns:
            dict: The classification with properly formatted explanation
        """
        if "llm_explanation" not in classification:
            return classification
            
        explanation = classification["llm_explanation"]
        
        # Check if the explanation already has the STEP format
        if not any(f"STEP {i}" in explanation for i in range(1, 6)) and not any(f"STEP {i}:" in explanation for i in range(1, 6)):
            domain_name = domain or "This domain"
            predicted_class = classification.get("predicted_class", "Unknown")
            is_service = classification.get("is_service_business", False)
            
            # Create a structured explanation with STEP format
            new_explanation = f"Based on the website content, {domain_name} is classified as a {predicted_class}\n\n"
            new_explanation += f"STEP 1: The website content provides sufficient information to analyze and classify the business, so the processing status is successful\n\n"
            new_explanation += f"STEP 2: The domain is not parked, under construction, or for sale, so it is not a Parked Domain\n\n"
            
            if is_service:
                confidence = int(classification.get('max_confidence', 0.8) * 100)
                new_explanation += f"STEP 3: The company is a service business that provides services to other businesses\n\n"
                new_explanation += f"STEP 4: Based on the service offerings described, this company is classified as a {predicted_class} with {confidence}% confidence\n\n"
                new_explanation += f"STEP 5: Since this is classified as a service business, the internal IT potential is set to 0/100\n\n"
            else:
                it_potential = classification.get("internal_it_potential", 50)
                new_explanation += f"STEP 3: The company is NOT a service/management business that provides ongoing IT or A/V services to clients\n\n"
                new_explanation += f"STEP 4: Since this is not a service business, we classify it as {predicted_class}\n\n"
                new_explanation += f"STEP 5: As a non-service business, we assess its internal IT potential at {it_potential}/100\n\n"
                
            # Include the original explanation as a summary
            new_explanation += f"In summary: {explanation}"
            classification["llm_explanation"] = new_explanation
            
        return classification
        
    def _check_special_domain_cases(self, domain: str, text_content: str) -> Optional[Dict[str, Any]]:
        """
        Check for special domain cases that need custom handling.
        
        Args:
            domain: The domain name
            text_content: The website content
            
        Returns:
            Optional[Dict[str, Any]]: Custom result if special case, None otherwise
        """
        domain_lower = domain.lower()
        
        # Check for special domains with known classifications
        # HostiFi - always MSP
        if "hostifi" in domain_lower:
            logger.info(f"Special case handling for known MSP domain: {domain}")
            return {
                "processing_status": 2,
                "is_service_business": True,
                "predicted_class": "Managed Service Provider",
                "internal_it_potential": 0,  # Set to 0 instead of None
                "confidence_scores": {
                    "Managed Service Provider": 85,
                    "Integrator - Commercial A/V": 8,
                    "Integrator - Residential A/V": 5,
                    "Corporate IT": 0  # Include Corporate IT with 0 score
                },
                "llm_explanation": f"{domain} is a cloud hosting platform specializing in Ubiquiti network management. They provide managed hosting services for UniFi Controller, UISP, and other network management software, which is a clear indication they are a Managed Service Provider focused on network infrastructure management.",
                "detection_method": "domain_override",
                "low_confidence": False,
                "max_confidence": 0.85
            }
            
        # Special handling for ciao.dk (known problematic vacation rental site)
        if domain_lower == "ciao.dk":
            logger.warning(f"Special handling for known vacation rental domain: {domain}")
            return {
                "processing_status": 2,
                "is_service_business": False,
                "predicted_class": "Non-Service Business",
                "internal_it_potential": 40,
                "confidence_scores": {
                    "Managed Service Provider": 5,
                    "Integrator - Commercial A/V": 3,
                    "Integrator - Residential A/V": 2,
                    "Corporate IT": 40  # Add Corporate IT score
                },
                "llm_explanation": f"{domain} appears to be a vacation rental/travel booking website offering holiday accommodations in various destinations. This is not a service business in the IT or A/V integration space. It's a travel industry business that might have a small internal IT department to maintain their booking systems and website.",
                "detection_method": "domain_override",
                "low_confidence": False,
                "max_confidence": 0.4
            }
            
        # Check for other vacation/travel-related domains
        vacation_terms = ["vacation", "holiday", "rental", "booking", "hotel", "travel", "accommodation", "ferie"]
        found_terms = [term for term in vacation_terms if term in domain_lower]
        if found_terms:
            logger.warning(f"Domain {domain} contains vacation/travel terms: {found_terms}")
            logger.warning(f"This domain should NOT be classified as Residential A/V")
            
            # Look for confirmation in the content
            travel_content_terms = ["booking", "accommodation", "stay", "vacation", "holiday", "rental"]
            if any(term in text_content.lower() for term in travel_content_terms):
                logger.warning(f"Content confirms {domain} is likely a travel/vacation site")
                return {
                    "processing_status": 2,
                    "is_service_business": False,
                    "predicted_class": "Non-Service Business",
                    "internal_it_potential": 35,
                    "confidence_scores": {
                        "Managed Service Provider": 5,
                        "Integrator - Commercial A/V": 3,
                        "Integrator - Residential A/V": 2,
                        "Corporate IT": 35  # Add Corporate IT score based on internal IT potential
                    },
                    "llm_explanation": f"{domain} appears to be a travel/vacation related business, not an IT or A/V service provider. The website focuses on accommodations, bookings, or vacation rentals rather than technology services or integration. This type of business might have minimal internal IT needs depending on its size.",
                    "detection_method": "domain_override",
                    "low_confidence": False,
                    "max_confidence": 0.35
                }
                
        # Check for transportation/logistics companies
        transport_terms = ["trucking", "transport", "logistics", "shipping", "freight", "delivery", "carrier"]
        found_transport_terms = [term for term in transport_terms if term in domain_lower]
        if found_transport_terms:
            logger.warning(f"Domain {domain} contains transportation terms: {found_transport_terms}")
            # Look for confirmation in the content
            transport_content_terms = ["shipping", "logistics", "fleet", "trucking", "transportation", "delivery"]
            if any(term in text_content.lower() for term in transport_content_terms):
                logger.warning(f"Content confirms {domain} is likely a transportation/logistics company")
                return {
                    "processing_status": 2,
                    "is_service_business": False,
                    "predicted_class": "Non-Service Business",
                    "internal_it_potential": 60,
                    "confidence_scores": {
                        "Managed Service Provider": 5,
                        "Integrator - Commercial A/V": 3,
                        "Integrator - Residential A/V": 2,
                        "Corporate IT": 60  # Add Corporate IT score based on internal IT potential
                    },
                    "llm_explanation": f"{domain} appears to be a transportation and logistics company, not an IT or A/V service provider. The website focuses on shipping, transportation, and logistics services rather than technology services or integration. This type of company typically has moderate internal IT needs to manage their operations and fleet management systems.",
                    "detection_method": "domain_override",
                    "low_confidence": False,
                    "max_confidence": 0.6
                }
                
        return None
    
    def _is_parked_domain(self, content: str) -> bool:
        """
        Enhanced detection of truly parked domains vs. just having minimal content.
        
        Args:
            content: The website content
            
        Returns:
            bool: True if the domain is parked/inactive
        """
        if not content:
            logger.info("Domain has no content at all, considering as parked")
            return True
            
        # Check for common parking phrases that indicate a domain is truly parked
        parking_phrases = [
            "domain is for sale", "buy this domain", "purchasing this domain", 
            "domain may be for sale", "this domain is for sale", "parked by",
            "domain parking", "this web page is parked", "website coming soon",
            "under construction", "site not published", "domain for sale",
            "under development", "this website is for sale"
        ]
        
        content_lower = content.lower()
        
        # Direct indicators of parked domains
        for phrase in parking_phrases:
            if phrase in content_lower:
                logger.info(f"Domain contains parking phrase: '{phrase}'")
                return True
        
        # Look for JavaScript-heavy sites with minimal crawled content
        if len(content.strip()) < 100:
            # More careful analysis before declaring parked
            
            # Check for common JS frameworks in the content
            js_indicators = ["react", "angular", "vue", "javascript", "script", "bootstrap", "jquery"]
            for indicator in js_indicators:
                if indicator in content_lower:
                    logger.info(f"Found JS framework indicator: {indicator} - may be JS-heavy site, not parked")
                    return False
            
            # Check for technical content that suggests a real site with poor crawling
            tech_indicators = ["<!doctype", "<html", "<head", "<meta", "<title", "<body", "<div"]
            tech_count = sum(1 for indicator in tech_indicators if indicator in content_lower)
            
            if tech_count >= 3:
                logger.info(f"Found {tech_count} HTML structure indicators - likely a real site with crawling issues")
                return False
            
            # Very little content with no indicators of real site structure
            if len(content.strip()) < 80:
                logger.info(f"Domain has extremely little content ({len(content.strip())} chars), considering as parked")
                return True
        
        # Very few words might indicate a parked domain, but be cautious
        words = re.findall(r'\b\w+\b', content_lower)
        unique_words = set(words)
        
        # An active site would typically have more unique words unless it's truly minimal
        if len(unique_words) < 15 and len(content.strip()) < 150:
            logger.info(f"Domain has very few unique words ({len(unique_words)}) and minimal content, considering as parked")
            return True
            
        return False
    
    def _create_process_did_not_complete_result(self, domain: str = None) -> Dict[str, Any]:
        """
        Create a standardized result for when processing couldn't complete.
        
        Args:
            domain: The domain name
            
        Returns:
            dict: Standardized process failure result
        """
        domain_name = domain or "Unknown domain"
        
        return {
            "processing_status": 0,
            "is_service_business": None,
            "predicted_class": "Process Did Not Complete",  # Always ensure a valid string
            "internal_it_potential": 0,  # Set to 0 instead of None
            "confidence_scores": {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0,
                "Corporate IT": 0  # Include Corporate IT with score 0
            },
            "llm_explanation": f"Classification process for {domain_name} could not be completed. This may be due to connection issues, invalid domain, or other technical problems.",
            "detection_method": "process_failed",
            "low_confidence": True,
            "max_confidence": 0.0  # Ensure this is set to 0, not 0.8
        }
    
    def _create_parked_domain_result(self, domain: str = None) -> Dict[str, Any]:
        """
        Create a standardized result for parked domains.
        
        Args:
            domain: The domain name
            
        Returns:
            dict: Standardized parked domain result
        """
        domain_name = domain or "This domain"
        
        return {
            "processing_status": 1,
            "is_service_business": None,
            "predicted_class": "Parked Domain",  # Always a valid string
            "internal_it_potential": 0,  # Set to 0 instead of None
            "confidence_scores": {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0,
                "Corporate IT": 0  # Include Corporate IT with score 0
            },
            "llm_explanation": f"{domain_name} appears to be a parked or inactive domain. No business-specific content was found to determine the company type. This may be a domain that is reserved but not yet in use, for sale, or simply under construction.",
            "detection_method": "parked_domain_detection",
            "low_confidence": True,
            "is_parked": True,  # Explicit flag for parked domains
            "max_confidence": 0.0
        }
            
    def detect_minimal_content(self, content: str) -> bool:
        """
        Detect if domain has minimal content.
        
        Args:
            content: The website content
            
        Returns:
            bool: True if the domain has minimal content
        """
        if not content or len(content.strip()) < 100:
            logger.info(f"Domain content is very short: {len(content) if content else 0} characters")
            return True
            
        # Count words in content
        words = re.findall(r'\b\w+\b', content.lower())
        unique_words = set(words)
        
        # Return true if few words or unique words
        if len(words) < 50:
            logger.info(f"Domain has few words ({len(words)}), likely minimal content")
            return True
            
        if len(unique_words) < 30:
            logger.info(f"Domain has few unique words ({len(unique_words)}), likely minimal content")
            return True
                
        return False
        
    def _extract_json(self, text: str) -> Optional[str]:
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
        
    def clean_json_string(self, json_str: str) -> str:
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
        
    def _validate_classification(self, classification: Dict[str, Any], domain: str = None) -> Dict[str, Any]:
        """
        Validate and normalize classification results.
        
        Args:
            classification: The classification to validate
            domain: Optional domain name for context
            
        Returns:
            dict: The validated classification
        """
        # Set default processing_status if not present
        if "processing_status" not in classification:
            classification["processing_status"] = 2  # Success
            
        # Check for parked domain
        if classification.get("processing_status") == 1:
            # This is a parked domain, no further validation needed
            classification["is_service_business"] = None
            classification["predicted_class"] = "Parked Domain"
            classification["internal_it_potential"] = 0  # Set to 0 instead of None
            classification["confidence_scores"] = {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0,
                "Corporate IT": 0  # Add Corporate IT with 0 score
            }
            classification["max_confidence"] = 0.0
            classification["low_confidence"] = True
            classification["is_parked"] = True
            return classification
            
        # Check for process failure
        if classification.get("processing_status") == 0:
            # Process did not complete, no further validation needed
            classification["is_service_business"] = None
            classification["predicted_class"] = "Process Did Not Complete"
            classification["internal_it_potential"] = 0  # Set to 0 instead of None
            classification["confidence_scores"] = {
                "Managed Service Provider": 0,
                "Integrator - Commercial A/V": 0,
                "Integrator - Residential A/V": 0,
                "Corporate IT": 0  # Add Corporate IT with 0 score
            }
            classification["max_confidence"] = 0.0
            classification["low_confidence"] = True
            return classification
        
        # Ensure required fields exist
        if "predicted_class" not in classification or classification["predicted_class"] is None:
            logger.warning("Missing or null predicted_class in classification, using fallback")
            
            # Check explanation for clues about what type of business this is
            explanation = classification.get("llm_explanation", "").lower()
            if "non-service business" in explanation or "not a service" in explanation:
                classification["predicted_class"] = "Non-Service Business"
            elif "travel" in explanation or "vacation" in explanation or "rental" in explanation:
                classification["predicted_class"] = "Non-Service Business"
            elif "transport" in explanation or "logistics" in explanation or "shipping" in explanation:
                classification["predicted_class"] = "Non-Service Business"
            else:
                classification["predicted_class"] = "Unknown"
            
            logger.info(f"Fixed null predicted_class to '{classification['predicted_class']}' based on explanation")
            
        if "is_service_business" not in classification:
            logger.warning("Missing is_service_business in classification, inferring from predicted_class")
            classification["is_service_business"] = classification["predicted_class"] in [
                "Managed Service Provider", 
                "Integrator - Commercial A/V", 
                "Integrator - Residential A/V"
            ]
            
        is_service = classification.get("is_service_business", True)
        
        # Check for very low confidence service business classifications
        if is_service and "confidence_scores" in classification:
            # Get highest confidence score
            highest_score = max(classification["confidence_scores"].values())
            if highest_score <= 15:
                # This is likely not actually a service business
                logger.warning(f"Very low confidence ({highest_score}) for service classification, recategorizing as Non-Service Business")
                classification["is_service_business"] = False
                classification["predicted_class"] = "Non-Service Business"
                is_service = False
                
                # Set appropriate internal IT potential
                if "llm_explanation" in classification:
                    # Extract potential internal IT score from explanation
                    it_match = re.search(r'internal IT.*?(\d+)[/\s]*100', classification["llm_explanation"])
                    if it_match:
                        classification["internal_it_potential"] = int(it_match.group(1))
                    else:
                        classification["internal_it_potential"] = 50  # Default medium value
        
        if "confidence_scores" not in classification:
            logger.warning("Missing confidence_scores in classification, using fallback")
            if is_service:
                classification["confidence_scores"] = {
                    "Managed Service Provider": 50,
                    "Integrator - Commercial A/V": 25,
                    "Integrator - Residential A/V": 15,
                    "Corporate IT": 0  # Add Corporate IT with 0 score
                }
            else:
                classification["confidence_scores"] = {
                    "Managed Service Provider": 5,
                    "Integrator - Commercial A/V": 3,
                    "Integrator - Residential A/V": 2
                }
                
        if "internal_it_potential" not in classification:
            logger.warning("Missing internal_it_potential in classification, using fallback")
            if is_service:
                classification["internal_it_potential"] = 0  # Set to 0 instead of None
            else:
                # Default middle value for unknown
                classification["internal_it_potential"] = 50
        
        if "llm_explanation" not in classification or not classification["llm_explanation"]:
            logger.warning("Missing llm_explanation in classification, using fallback")
            if is_service:
                classification["llm_explanation"] = f"Based on the available information, this appears to be a {classification['predicted_class']}."
            else:
                classification["llm_explanation"] = f"This appears to be a non-service business. It doesn't provide IT or A/V integration services."
        
        # Normalize confidence scores
        confidence_scores = classification["confidence_scores"]
        
        # Check if scores need to be converted from 0-1 to 1-100 scale
        if any(isinstance(score, float) and 0 <= score <= 1 for score in confidence_scores.values()):
            logger.info("Converting confidence scores from 0-1 scale to 1-100")
            confidence_scores = {k: int(v * 100) for k, v in confidence_scores.items()}
        
        # Ensure all required categories exist
        required_categories = ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V"]
        for category in required_categories:
            if category not in confidence_scores:
                logger.warning(f"Missing category {category} in confidence scores, adding default")
                confidence_scores[category] = 5 if not is_service else 30
                
        # Ensure scores are within valid range (1-100)
        confidence_scores = {k: max(1, min(100, int(v))) for k, v in confidence_scores.items()}
        
        # For non-service businesses, ensure service scores are appropriately low
        if not is_service:
            for category in required_categories:
                if confidence_scores[category] > 10:
                    logger.warning(f"Reducing {category} score for non-service business")
                    confidence_scores[category] = min(confidence_scores[category], 10)
            
            # Ensure internal_it_potential is an integer
            if classification["internal_it_potential"] is not None:
                classification["internal_it_potential"] = int(classification["internal_it_potential"])
                
            # Add Corporate IT score based on internal_it_potential
            it_potential = classification.get("internal_it_potential", 50)
            confidence_scores["Corporate IT"] = it_potential
        else:
            # Add Corporate IT score with value 0 for service businesses
            confidence_scores["Corporate IT"] = 0
                
        # For service businesses, ensure scores are differentiated
        if is_service and (len(set(confidence_scores.values())) <= 1 or 
                            max(confidence_scores.values()) - min(confidence_scores.values()) < 5):
            logger.warning("Confidence scores not sufficiently differentiated for service business, adjusting them")
            
            pred_class = classification["predicted_class"]
            
            # Set base scores to ensure strong differentiation
            if pred_class == "Managed Service Provider":
                confidence_scores = {
                    "Managed Service Provider": 80,
                    "Integrator - Commercial A/V": 15,
                    "Integrator - Residential A/V": 5,
                    "Corporate IT": 0  # Ensure Corporate IT is always included with 0 for service businesses
                }
            elif pred_class == "Integrator - Commercial A/V":
                confidence_scores = {
                    "Integrator - Commercial A/V": 80,
                    "Managed Service Provider": 15,
                    "Integrator - Residential A/V": 5,
                    "Corporate IT": 0  # Ensure Corporate IT is always included with 0 for service businesses
                }
            else:  # Residential A/V
                confidence_scores = {
                    "Integrator - Residential A/V": 80,
                    "Integrator - Commercial A/V": 15,
                    "Managed Service Provider": 5,
                    "Corporate IT": 0  # Ensure Corporate IT is always included with 0 for service businesses
                }
        
        # For service businesses, ensure predicted class matches highest confidence category
        if is_service:
            highest_category = max(confidence_scores.items(), key=lambda x: x[1] if x[0] != "Corporate IT" else 0)[0]
            if classification["predicted_class"] != highest_category:
                logger.warning(f"Predicted class {classification['predicted_class']} doesn't match highest confidence category {highest_category}, fixing")
                classification["predicted_class"] = highest_category
            
        # Calculate max confidence for consistency
        if is_service:
            classification["max_confidence"] = confidence_scores[classification["predicted_class"]] / 100.0
        else:
            # For non-service businesses, max confidence is based on internal IT potential certainty
            classification["max_confidence"] = 0.8 if classification["internal_it_potential"] is not None else 0.5
        
        # Add low_confidence flag based on highest score or other factors
        if is_service:
            classification["low_confidence"] = confidence_scores[classification["predicted_class"]] < 40
        else:
            # For non-service, we're less confident overall
            classification["low_confidence"] = True
            
        # Update the classification with validated scores
        classification["confidence_scores"] = confidence_scores
        
        return classification
        
    def _parse_free_text(self, text: str, domain: str = None) -> Dict[str, Any]:
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
            return self._create_parked_domain_result(domain)
            
        # Check for process failure indicators
        if any(phrase in text_lower for phrase in ["process did not complete", "couldn't process", "failed to process"]):
            return self._create_process_did_not_complete_result(domain)
        
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
            predicted_class = "Non-Service Business"
            
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
            msp_score = sum(1 for keyword in self.msp_indicators if keyword in text_lower)
            commercial_score = sum(1 for keyword in self.commercial_av_indicators if keyword in text_lower)
            residential_score = sum(1 for keyword in self.residential_av_indicators if keyword in text_lower)
            
            total_score = max(1, msp_score + commercial_score + residential_score)
            
            # Calculate proportional scores
            msp_conf = 0.30 + (0.5 * msp_score / total_score) if msp_score > 0 else 0.08
            comm_conf = 0.30 + (0.5 * commercial_score / total_score) if commercial_score > 0 else 0.08
            resi_conf = 0.30 + (0.5 * residential_score / total_score) if residential_score > 0 else 0.08
            
            confidence_scores = {
                "Managed Service Provider": int(msp_conf * 100),
                "Integrator - Commercial A/V": int(comm_conf * 100),
                "Integrator - Residential A/V": int(resi_conf * 100),
                "Corporate IT": 0  # Service businesses always have 0 for Corporate IT
            }
            
            # Ensure predicted class has highest confidence
            if predicted_class in confidence_scores:
                highest_score = max(confidence_scores.items(), key=lambda x: x[1] if x[0] != "Corporate IT" else 0)[0]
                confidence_scores[predicted_class] = max(confidence_scores[predicted_class], confidence_scores[highest_score] + 5)
        else:
            # Low confidence scores for all service categories
            confidence_scores = {
                "Managed Service Provider": 5,
                "Integrator - Commercial A/V": 3,
                "Integrator - Residential A/V": 2,
                "Corporate IT": internal_it_potential or 30  # Add Corporate IT score
            }
            
        # Extract or generate explanation
        explanation = self._extract_explanation(text)
        if not explanation or len(explanation) < 100:
            explanation = self._generate_explanation(predicted_class, domain, is_service, internal_it_potential)
            
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
            "detection_method": "text_parsing",
            "low_confidence": is_service and max_confidence < 0.4,
            "max_confidence": max_confidence
        }
        
    def _extract_explanation(self, text: str) -> str:
        """
        Extract explanation from text.
        
        Args:
            text: The text to extract explanation from
            
        Returns:
            str: The extracted explanation
        """
        # First try to find explanation directly
        explanation_patterns = [
            r'explanation[:\s]+([^}{"]*)',
            r'based on[^.]*(?:[^.]*\.){2,5}',
            r'(?:the company|the website|the text|the content|this appears)[^.]*(?:[^.]*\.){2,5}'
        ]
        
        for pattern in explanation_patterns:
            matches = re.search(pattern, text, re.IGNORECASE)
            if matches:
                explanation = matches.group(0) if pattern.startswith('based') or pattern.startswith('(?:') else matches.group(1)
                # Clean up the explanation
                explanation = explanation.replace('explanation:', '').replace('explanation', '').strip()
                
                if len(explanation) > 50:
                    return explanation
                    
        # If still no good explanation, take the longest sentence group
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if sentences:
            longest_group = ""
            for i in range(len(sentences) - 2):
                group = " ".join(sentences[i:i+3])
                if len(group) > len(longest_group) and "confidence" not in group.lower() and "score" not in group.lower():
                    longest_group = group
                    
            if len(longest_group) > 50:
                return longest_group
                
        return ""
        
    def _generate_explanation(self, predicted_class: str, domain: str = None, 
                              is_service: bool = True, internal_it_potential: int = None) -> str:
        """
        Generate explanation based on predicted class and service status.
        
        Args:
            predicted_class: The predicted class
            domain: Optional domain name
            is_service: Whether this is a service business
            internal_it_potential: Internal IT potential score
            
        Returns:
            str: The generated explanation
        """
        domain_name = domain or "The company"
        
        if is_service:
            if predicted_class == "Managed Service Provider":
                return f"{domain_name} appears to be a Managed Service Provider (MSP) based on the available evidence. The content suggests a focus on IT services, technical support, and technology management for business clients. MSPs typically provide services like network management, cybersecurity, cloud solutions, and IT infrastructure support. Since this is a service business, the internal IT potential is set to 0/100."
                
            elif predicted_class == "Integrator - Commercial A/V":
                return f"{domain_name} appears to be a Commercial A/V Integrator based on the available evidence. The content suggests a focus on designing and implementing audiovisual solutions for businesses, such as conference rooms, digital signage, and professional audio systems for commercial environments. Since this is a service business, the internal IT potential is set to 0/100."
                
            elif predicted_class == "Integrator - Residential A/V":
                return f"{domain_name} appears to be a Residential A/V Integrator based on the available evidence. The content suggests a focus on home automation, smart home technology, and audiovisual systems for residential clients, such as home theaters and whole-house audio solutions. Since this is a service business, the internal IT potential is set to 0/100."
        else:
            # Non-service business explanation
            if internal_it_potential is not None:
                it_level = "significant" if internal_it_potential > 70 else \
                           "moderate" if internal_it_potential > 40 else "minimal"
                           
                return f"{domain_name} does not appear to be a service business in the IT or A/V integration space. It is not providing managed services, IT support, or audio/visual integration to clients. Rather, it appears to be a business that might have {it_level} internal IT needs of its own. The internal IT potential is assessed at {internal_it_potential}/100."
            else:
                return f"{domain_name} does not appear to be a service business in the IT or A/V integration space. There's no evidence that it provides managed services, IT support, or audio/visual integration to clients."
            
        return f"Based on the available information, {domain_name} appears to be a {predicted_class}."
        
    def _fallback_classification(self, text_content: str, domain: str = None) -> Dict[str, Any]:
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
        service_count = sum(1 for term in self.service_business_indicators if term in text_lower)
        
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
            msp_count = sum(1 for keyword in self.msp_indicators if keyword in text_lower)
            commercial_count = sum(1 for keyword in self.commercial_av_indicators if keyword in text_lower)
            residential_count = sum(1 for keyword in self.residential_av_indicators if keyword in text_lower)
            
            total_count = max(1, msp_count + commercial_count + residential_count)
            
            # Check for negative indicators
            for indicator, neg_class in self.negative_indicators.items():
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
                        return self._create_non_service_result(domain, text_content)
            
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
                    return self._create_non_service_result(domain, text_content)
                    
            # Additional content checks for non-service businesses
            if "transport" in text_lower or "logistics" in text_lower or "shipping" in text_lower or "trucking" in text_lower:
                # This is likely a transportation/logistics company, not a service provider
                logger.info("Content suggests transportation/logistics business, reclassifying as non-service")
                return self._create_non_service_result(domain, text_content)
                
            # Determine predicted class based on highest confidence
            predicted_class = max(confidence.items(), key=lambda x: x[1])[0]
            
            # Apply the adjustment logic to ensure meaningful differences between categories
            if predicted_class == "Managed Service Provider" and confidence["Managed Service Provider"] > 0.5:
                confidence["Integrator - Residential A/V"] = min(confidence["Integrator - Residential A/V"], 0.12)
            
            # Generate explanation
            explanation = self._generate_explanation(predicted_class, domain, is_service)
            explanation += " (Note: This classification is based on our fallback system, as detailed analysis was unavailable.)"
            
            # Convert decimal confidence to 1-100 range
            confidence_scores = {k: int(v * 100) for k, v in confidence.items()}
            
            # Add Corporate IT with score 0 for service businesses
            confidence_scores["Corporate IT"] = 0
            
            return {
                "processing_status": 2,  # Success
                "is_service_business": True,
                "predicted_class": predicted_class,
                "internal_it_potential": 0,  # Set to 0 for service businesses
                "confidence_scores": confidence_scores,
                "llm_explanation": explanation,
                "detection_method": "fallback",
                "low_confidence": True,
                "max_confidence": confidence_scores[predicted_class] / 100.0
            }
        else:
            return self._create_non_service_result(domain, text_content)
    
    def _create_non_service_result(self, domain: str, text_content: str) -> Dict[str, Any]:
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
            explanation = f"{domain or 'This company'} appears to be a transportation and logistics service provider, not a service/management business or an A/V integrator. The company focuses on physical transportation, shipping, or logistics rather than providing IT or A/V services to clients. This type of company typically has moderate to significant internal IT needs to manage their operations, logistics systems, and fleet management."
        elif "vacation" in text_lower or "hotel" in text_lower or "accommodation" in text_lower or "booking" in text_lower:
            explanation = f"{domain or 'This company'} appears to be in the travel, tourism, or hospitality industry, not an IT service provider or A/V integrator. The company focuses on providing accommodations, vacation rentals, or travel-related services rather than IT or A/V services. This type of business typically has low to moderate internal IT needs to maintain booking systems and websites."
        elif "retail" in text_lower or "shop" in text_lower or "store" in text_lower or "product" in text_lower:
            explanation = f"{domain or 'This company'} appears to be a retail or e-commerce business, not an IT service provider or A/V integrator. The company sells products rather than providing IT or A/V services to clients. This type of business typically has varying levels of internal IT needs depending on their size and online presence."
        else:
            explanation = f"{domain or 'This company'} does not appear to be a service business in the IT or A/V integration space. There is no evidence that it provides managed services, IT support, or audio/visual integration to clients. Rather, it appears to be a company that might have its own internal IT needs (estimated potential: {internal_it_potential}/100)."
            
        # Create non-service business result with minimal confidence scores for service categories
        return {
            "processing_status": 2,  # Success
            "is_service_business": False,
            "predicted_class": "Non-Service Business",
            "internal_it_potential": internal_it_potential,
            "confidence_scores": {
                "Managed Service Provider": 5,
                "Integrator - Commercial A/V": 3,
                "Integrator - Residential A/V": 2,
                "Corporate IT": internal_it_potential  # Add Corporate IT score based on internal IT potential
            },
            "llm_explanation": explanation,
            "detection_method": "non_service_detection",
            "low_confidence": True,
            "max_confidence": 0.8  # Reasonably confident in non-service classification
        }
        
    def check_confidence_alignment(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if the confidence scores align with the predicted class.
        
        Args:
            classification: The classification to check
            
        Returns:
            dict: The updated classification with aligned scores
        """
        # Only relevant for service businesses
        if not classification.get("is_service_business", False):
            return classification
            
        if "confidence_scores" not in classification or "predicted_class" not in classification:
            return classification
            
        confidence_scores = classification["confidence_scores"]
        predicted_class = classification["predicted_class"]
        
        # Make sure predicted class has the highest score
        if predicted_class in confidence_scores:
            highest_score = max(confidence_scores.items(), key=lambda x: x[1] if x[0] != "Corporate IT" else 0)[0]
            
            if highest_score != predicted_class:
                logger.warning(f"Predicted class {predicted_class} doesn't match highest confidence {highest_score}, adjusting")
                # Find current score of predicted class
                current_score = confidence_scores[predicted_class]
                # Boost it above the highest
                confidence_scores[predicted_class] = confidence_scores[highest_score] + 5
                
        # Make sure scores are differentiated
        service_scores = {k: v for k, v in confidence_scores.items() if k != "Corporate IT"}
        if len(set(service_scores.values())) <= 1 or max(service_scores.values()) - min(service_scores.values()) < 5:
            logger.warning("Confidence scores not differentiated enough, adjusting")
            
            # Set up differentiated scores
            if predicted_class == "Managed Service Provider":
                confidence_scores.update({
                    "Managed Service Provider": 80,
                    "Integrator - Commercial A/V": 15,
                    "Integrator - Residential A/V": 5
                })
            elif predicted_class == "Integrator - Commercial A/V":
                confidence_scores.update({
                    "Integrator - Commercial A/V": 80,
                    "Managed Service Provider": 15,
                    "Integrator - Residential A/V": 5
                })
            else:  # Residential A/V
                confidence_scores.update({
                    "Integrator - Residential A/V": 80,
                    "Integrator - Commercial A/V": 15,
                    "Managed Service Provider": 5
                })
                
        # Ensure Corporate IT is set to 0 for service businesses
        confidence_scores["Corporate IT"] = 0
                
        # Update classification
        classification["confidence_scores"] = confidence_scores
        return classification
    
    def process_domain(self, domain: str, content: str = None) -> Dict[str, Any]:
        """
        Comprehensive domain processing pipeline.
        
        Args:
            domain: The domain to process
            content: Optional pre-fetched content
            
        Returns:
            dict: The processed classification result
        """
        logger.info(f"Processing domain: {domain}")
        
        if not content:
            logger.warning(f"No content provided for {domain}")
            return self._create_process_did_not_complete_result(domain)
            
        # Check for special domain cases first
        special_result = self._check_special_domain_cases(domain, content)
        if special_result:
            logger.info(f"Using special case handling for {domain}")
            return special_result
            
        # Normal classification pipeline
        result = self.classify(content, domain)
        
        # Ensure confidence scores are aligned and differentiated
        result = self.check_confidence_alignment(result)
        
        # Set low confidence flag if appropriate
        max_confidence = result.get("max_confidence", 0.5)
        if result.get("is_service_business", False):
            result["low_confidence"] = max_confidence < 0.4
        else:
            result["low_confidence"] = max_confidence < 0.6
            
        logger.info(f"Completed processing for {domain}: {result['predicted_class']}")
        return result
