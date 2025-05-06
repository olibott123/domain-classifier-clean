"""Prompt builder for LLM-based domain classification."""
import logging
import os
import csv
from typing import Dict, List, Any

# Set up logging
logger = logging.getLogger(__name__)

def load_examples_from_knowledge_base() -> Dict[str, List[Dict[str, str]]]:
    """
    Load examples from knowledge base CSV file with enhanced logging.
    
    Returns:
        Dict mapping categories to lists of example domains with content
    """
    examples = {
        "Managed Service Provider": [],
        "Integrator - Commercial A/V": [],
        "Integrator - Residential A/V": [],
        "Internal IT Department": []
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
                        # Handle the old name if it still exists in CSV
                        if company_type == "Non-Service Business":
                            company_type = "Internal IT Department"
                            
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
                        # Handle the old name if it still exists in CSV
                        if company_type == "Non-Service Business":
                            company_type = "Internal IT Department"
                            
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
            else:  # Internal IT Department
                examples[category].append({
                    'domain': 'example-nonservice.com',
                    'content': 'We are an online retailer selling consumer electronics and accessories. Browse our wide selection of products for your home and office needs. Shop now for great deals on computers, phones, and entertainment gadgets with fast shipping to your door.'
                })
    
    return examples

def build_decision_tree_prompt(examples: Dict[str, List[Dict[str, str]]]) -> str:
    """
    Build a prompt that implements the decision tree approach.
    
    Args:
        examples: Knowledge base examples
        
    Returns:
        str: The system prompt for the LLM
    """
    # Implement the decision tree prompt
    system_prompt = """You are an expert business analyst who specializes in categorizing companies based on website content.
You need to follow a specific decision tree to classify the business:

STEP 1: Determine if processing can complete
- If there is insufficient content to analyze, output "Process Did Not Complete"

STEP 2: Check if this is a parked/inactive domain
- If the domain is parked, under construction, or for sale, classify as "Parked Domain"

STEP 3: Determine if the company is a TECHNOLOGY SERVICE/MANAGEMENT business:
- Technology Service businesses provide technology-focused services, IT support, or managed technology solutions to clients
- They must specifically offer technology, IT, A/V, or security/safety system services as their core business
- Examples: IT service providers, cybersecurity firms, network management companies, A/V integrators, security system integrators

IMPORTANT: Many non-technology businesses provide "services" to customers (healthcare, hospitality, aged care, etc.) but are NOT technology service providers. These should be classified as "Internal IT Department" even if they use the word "service" in their description.

STEP 4: If it IS a technology service business, classify into ONE of these specific categories based on the following clear distinctions:

1. Managed Service Provider (MSP):
   - Primary focus: IT infrastructure management, network support, software management, cloud hosting
   - Specific services: Help desk support, remote IT monitoring, network administration, server management
   - Service model: Typically ongoing recurring services with monthly/annual contracts, service level agreements (SLAs)
   - Security focus: Cybersecurity, digital security, data protection, IT security consulting
   - NOT included: Physical security systems, hardware installation as primary business, one-time projects

2. Integrator - Commercial A/V:
   - Primary focus: Design, installation, and integration of physical technology systems for commercial environments
   - Specific systems: Fire alarm systems, security cameras, access control, physical surveillance, conference rooms
   - Service model: Project-based installations and equipment maintenance, often with maintenance contracts
   - Security focus: Physical security systems, fire safety, building access control, CCTV
   - Key examples: Fire and security system companies, commercial audio/video installers, building automation integrators

3. Integrator - Residential A/V:
   - Primary focus: Home automation, entertainment systems, residential security for individual homeowners
   - Specific systems: Home theater, smart home, residential surveillance, whole-house audio
   - Service model: Custom installations for homes and high-end residences
   - Security focus: Home security systems, residential cameras, smart locks, alarm systems
   - NOT included: Commercial building technologies or office-focused solutions

CRITICAL DISTINCTION - SECURITY SERVICES:
- Physical security companies (fire alarms, security cameras, access control systems) = Commercial A/V Integrator
- Cybersecurity companies (network security, data protection, digital threat detection) = Managed Service Provider

STEP 5: If it is NOT a technology service business, determine if it's a business that could have an internal IT department:
- Medium to large businesses with multiple employees and locations typically have internal IT
- Enterprises, corporations, manufacturers, retailers, healthcare providers, financial institutions, nonprofit organizations, churches, etc.
- These are organizations that USE technology but don't PROVIDE technology services to clients

IMPORTANT GUIDELINES FOR CLASSIFICATION:
- Technology service businesses should have an Internal IT Department score of 0 (as they are providing technology services, not consuming IT internally)
- Non-technology service businesses should have an Internal IT Department score between 1-100 indicating their internal IT potential
- Healthcare organizations, retirement homes, aged care facilities, hospitality businesses are NOT technology service providers, even if they offer "services" to clients
- Churches, religious organizations, educational institutions, and nonprofits are almost always "Internal IT Department"
- Security system integrators and fire alarm companies that primarily install and integrate physical systems in commercial buildings are "Commercial A/V Integrators", not MSPs
- Cybersecurity companies that focus on digital/network security services are MSPs, not Commercial Integrators
- Companies that focus on ongoing IT infrastructure management and network support are MSPs
- Companies that primarily design and install integrated physical systems in buildings are Commercial Integrators
- Companies that primarily serve residential clients with home systems are Residential Integrators
- Vacation rental services, travel agencies, and hospitality businesses are NOT technology service providers
- Web design agencies and general marketing firms are NOT typically MSPs unless they explicitly offer ongoing IT services
- Media production companies are NOT necessarily A/V integrators
- Purely online retailers or e-commerce sites generally don't provide technology services and are NOT MSPs
- Transportation and logistics companies are NOT technology service providers

Here are examples of correctly classified companies:"""

    # Add examples to the system prompt
    for category in ["Managed Service Provider", "Integrator - Commercial A/V", "Integrator - Residential A/V", "Internal IT Department"]:
        if category in examples and examples[category]:
            system_prompt += f"\n\n## {category} examples:\n"
            for example in examples[category][:2]:  # Just use 2 examples per category
                snippet = example.get('content', '')[:300].replace('\n', ' ').strip()
                system_prompt += f"\nDomain: {example.get('domain', 'example.com')}\nContent snippet: {snippet}...\nClassification: {category}\n"
    
    # Add instructions for output format with enhanced company description requirement
    system_prompt += """\n\nYou MUST provide your analysis in JSON format with the following structure:
{
  "processing_status": [0 if processing failed, 1 if domain is parked, 2 if classification successful],
  "is_service_business": [true/false - whether this is a TECHNOLOGY service/management business - NOT just any service business],
  "predicted_class": [if is_service_business=true: "Managed Service Provider", "Integrator - Commercial A/V", or "Integrator - Residential A/V" | if is_service_business=false: "Internal IT Department" | if processing_status=0: "Process Did Not Complete" | if processing_status=1: "Parked Domain"],
  "internal_it_potential": [if is_service_business=false: score from 1-100 indicating likelihood the business could have internal IT department | if is_service_business=true: 0],
  "company_description": [A clear, detailed paragraph (75-100 words) describing what the company actually does. Include specific services, target markets, approach, and any distinctive attributes. Be specific rather than generic.],
  "confidence_scores": {
    "Integrator - Commercial A/V": [Integer from 1-100, only relevant if is_service_business=true],
    "Integrator - Residential A/V": [Integer from 1-100, only relevant if is_service_business=true],
    "Managed Service Provider": [Integer from 1-100, only relevant if is_service_business=true],
    "Internal IT Department": [0 if is_service_business=true, integer from 1-100 if is_service_business=false]
  },
  "llm_explanation": "A detailed explanation of your reasoning and classification process"
}

IMPORTANT INSTRUCTIONS:
1. You MUST follow the decision tree exactly as specified.
2. For technology service businesses, provide DIFFERENT confidence scores for each category.
3. For technology service businesses, the Internal IT Department score MUST be 0.
4. For non-technology service businesses, all service-type confidence scores should be very low (1-10).
5. Internal IT potential should be 0 for technology service businesses and scored 1-100 for non-technology service businesses.
6. Your llm_explanation must detail your decision process through each step.
7. Mention specific evidence from the text that supports your classification.
8. Healthcare providers, aged care facilities, nursing homes, churches, and educational organizations should always be classified as Internal IT Department.
9. Your explanation MUST be formatted with STEP 1, STEP 2, etc. clearly labeled for each stage of the decision tree.
10. The company_description should be 1-2 substantive paragraphs (75-100 words) focusing on what the company actually does, their specific services or products, target markets, and unique attributes. Avoid vague, generic descriptions.
11. REMEMBER: Physical security companies (fire alarms, cameras, access control) are Commercial Integrators; cybersecurity companies (data protection, network security) are MSPs.

YOUR RESPONSE MUST BE A SINGLE VALID JSON OBJECT WITH NO OTHER TEXT BEFORE OR AFTER."""

    return system_prompt
