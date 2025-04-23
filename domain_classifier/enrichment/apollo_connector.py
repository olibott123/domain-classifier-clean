"""Apollo.io connector for company data enrichment."""
import requests
import logging
import os
from typing import Dict, Any, Optional

# Set up logging
logger = logging.getLogger(__name__)

class ApolloConnector:
    def __init__(self, api_key: str = None):
        """
        Initialize the Apollo connector with API key.
        
        Args:
            api_key: The API key for Apollo.io
        """
        self.api_key = api_key or os.environ.get("APOLLO_API_KEY")
        if not self.api_key:
            logger.warning("No Apollo API key provided. Enrichment will not be available.")
        
        self.base_url = "https://api.apollo.io/v1"
    
    def enrich_company(self, domain: str) -> Optional[Dict[str, Any]]:
        """
        Enrich a company profile using Apollo.io.
        
        Args:
            domain: The domain to enrich
            
        Returns:
            dict: The enriched company data or None if failed
        """
        if not self.api_key:
            logger.warning(f"Cannot enrich {domain}: No Apollo API key")
            return None
            
        try:
            # Apollo's organizations/enrich endpoint
            endpoint = f"{self.base_url}/organizations/enrich"
            
            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache"
            }
            
            payload = {
                "api_key": self.api_key,
                "domain": domain
            }
            
            response = requests.post(endpoint, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('organization'):
                    logger.info(f"Successfully enriched {domain} with Apollo data")
                    return self._format_company_data(data['organization'])
                else:
                    logger.warning(f"No organization data found for {domain}")
            else:
                logger.warning(f"Apollo API returned status {response.status_code} for {domain}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error enriching company with Apollo: {e}")
            return None
    
    def _format_company_data(self, apollo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format Apollo data into a standardized structure.
        
        Args:
            apollo_data: The raw Apollo organization data
            
        Returns:
            dict: Formatted company data
        """
        # Extract and format the most relevant fields
        return {
            "name": apollo_data.get('name'),
            "website": apollo_data.get('website_url'),
            "industry": apollo_data.get('industry'),
            "employee_count": apollo_data.get('estimated_num_employees'),
            "revenue": apollo_data.get('estimated_annual_revenue'),
            "founded_year": apollo_data.get('founded_year'),
            "linkedin_url": apollo_data.get('linkedin_url'),
            "phone": apollo_data.get('phone'),
            "address": self._format_address(apollo_data),
            "technologies": apollo_data.get('technologies', []),
            "funding": {
                "total_funding": apollo_data.get('total_funding'),
                "latest_funding_round": apollo_data.get('latest_funding_round'),
                "latest_funding_amount": apollo_data.get('latest_funding_amount')
            }
        }
    
    def _format_address(self, apollo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format address data from Apollo."""
        return {
            "street": apollo_data.get('street_address'),
            "city": apollo_data.get('city'),
            "state": apollo_data.get('state'),
            "country": apollo_data.get('country'),
            "zip": apollo_data.get('postal_code')
        }
