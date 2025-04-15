def get_indicators():
    """
    Return dictionaries of keyword indicators for different company types.
    
    Returns:
        dict: Dictionary containing indicator lists for each company type
    """
    indicators = {
        "msp": [
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
            "hosting", "hostifi", "managed hosting", "cloud hosting", "vcio",
            "it consulting", "tech consulting", "virtual cio", "vdm", "virtual dm", 
            "it outsourcing", "it consulting", "it solutions", "computer repair",
            "technical services", "information technology", "service desk"
        ],
        
        "commercial_av": [
            "commercial integration", "av integration", "audio visual", "audiovisual",
            "conference room", "meeting room", "digital signage", "video wall",
            "commercial audio", "commercial display", "projection system", "projector",
            "commercial automation", "room scheduling", "presentation system", "boardroom",
            "professional audio", "business audio", "commercial installation", "enterprise",
            "huddle room", "training room", "av design", "control system", "av consultant",
            "crestron", "extron", "biamp", "amx", "polycom", "cisco", "zoom room",
            "teams room", "corporate", "business communication", "commercial sound",
            "av solutions", "video conferencing", "professional sound", "broadcast",
            "pro av", "a/v", "a/v systems", "a/v solutions", "audiovisual systems",
            "commercial integration", "business presentation", "corporate communication"
        ],
        
        "residential_av": [
            "home automation", "smart home", "home theater", "residential integration",
            "home audio", "home sound", "custom installation", "home control", "home cinema",
            "residential av", "whole home audio", "distributed audio", "multi-room",
            "lighting control", "home network", "home wifi", "entertainment system",
            "sonos", "control4", "savant", "lutron", "residential automation", "smart tv",
            "home entertainment", "consumer", "residential installation", "home integration",
            "whole house", "home theater system", "home automation", "smart lighting",
            "residential", "custom", "custom home", "home security", "home theater design",
            "home cinema installation", "residential audio", "residential sound", "homeowners"
        ],
        
        "service_business": [
            "services", "solutions", "consulting", "provider", "professional services",
            "managed", "support", "experts", "professionals", "specialists", "agency",
            "firm", "consultancy", "advisory", "helping", "serving", "assistance",
            "partner", "maintenance", "24/7", "service-based", "management",
            "outsourced", "service provider", "support plan", "routine service",
            "ongoing service", "service package", "subscription", "service tier"
        ],
        
        "internal_it": [
            "enterprise", "corporation", "staff", "team", "employees", "personnel",
            "department", "division", "it department", "it team", "it staff",
            "corporate", "headquarters", "company", "business", "organization",
            "locations", "branches", "offices", "global", "nationwide", "workforce",
            "operations", "manufacturing", "production", "factory", "retail",
            "multiple locations", "in-house", "internal", "it manager", "cio", "cto"
        ],
        
        "negative": {
            # Vacation rental indicators (should NOT be classified as Residential A/V)
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
            
            # E-commerce indicators (should NOT be classified as MSP)
            "add to cart": "NOT_MSP",
            "add to basket": "NOT_MSP",
            "shopping cart": "NOT_MSP",
            "free shipping": "NOT_MSP",
            "checkout": "NOT_MSP",
            
            # Generic business indicators (should NOT be classified as service business)
            "our products": "NOT_SERVICE",
            "product catalog": "NOT_SERVICE",
            "manufacturer": "NOT_SERVICE",
            "retailer": "NOT_SERVICE",
            "e-commerce": "NOT_SERVICE",
            "buy now": "NOT_SERVICE"
        }
    }
    
    return indicators
