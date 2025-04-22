"""
Keyword indicators for domain classification.
This module contains lists of indicator terms for different company types.
"""

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
