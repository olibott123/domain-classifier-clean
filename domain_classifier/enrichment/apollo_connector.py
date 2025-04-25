"""Apollo.io connector for company data enrichment."""
import requests
import logging
import os
import json
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
            
            # Use the X-Api-Key header as required by Apollo
            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "X-Api-Key": self.api_key  # This is the correct header format
            }
            
            payload = {
                "domain": domain
            }
            
            logger.info(f"Attempting to enrich data for domain: {domain}")
            
            response = requests.post(endpoint, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('organization'):
                    logger.info(f"Successfully enriched {domain} with Apollo data")
                    return self._format_company_data(data['organization'])
                else:
                    logger.warning(f"Apollo API returned 200 but no organization data for {domain}")
                    logger.debug(f"Apollo API response: {json.dumps(data)[:500]}...")
            else:
                # Enhanced error logging
                try:
                    error_details = response.json()
                    logger.error(f"Apollo API error ({response.status_code}) for {domain}: {error_details}")
                except:
                    logger.error(f"Apollo API error ({response.status_code}) for {domain}: {response.text[:500]}")
                
            return None
            
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while connecting to Apollo API for {domain}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error while connecting to Apollo API for {domain}")
            return None
        except Exception as e:
            logger.error(f"Error enriching company with Apollo: {e}")
            return None
    
    def _format_company_data(self, apollo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format Apollo data into a standardized structure with enhanced location data.
        
        Args:
            apollo_data: The raw Apollo organization data
            
        Returns:
            dict: Formatted company data
        """
        try:
            # Log a sample of the received data for debugging
            logger.debug(f"Sample of Apollo data: {json.dumps({k: apollo_data.get(k) for k in list(apollo_data.keys())[:5]})}")
            
            # Extract and format location data
            location_data = {
                "street": apollo_data.get('street_address'),
                "city": apollo_data.get('city'),
                "state": apollo_data.get('state'),
                "country": apollo_data.get('country'),
                "postal_code": apollo_data.get('postal_code'),
                "formatted_address": None  # Will fill below if components exist
            }
            
            # Create a formatted address if we have enough components
            address_parts = []
            if location_data["street"]:
                address_parts.append(location_data["street"])
            if location_data["city"]:
                address_parts.append(location_data["city"])
            if location_data["state"]:
                address_parts.append(location_data["state"])
            if location_data["postal_code"]:
                address_parts.append(location_data["postal_code"])
            if location_data["country"]:
                address_parts.append(location_data["country"])
                
            if address_parts:
                location_data["formatted_address"] = ", ".join(address_parts)
                
            # Also add coordinates if available
            if apollo_data.get('latitude') and apollo_data.get('longitude'):
                location_data["coordinates"] = {
                    "latitude": apollo_data.get('latitude'),
                    "longitude": apollo_data.get('longitude')
                }
            
            # Extract phone number safely
            phone = apollo_data.get('phone', '')
            if not phone and apollo_data.get('phone_numbers') and len(apollo_data.get('phone_numbers', [])) > 0:
                phone = apollo_data.get('phone_numbers')[0]
                
            # Extract and format the most relevant fields
            return {
                "name": apollo_data.get('name'),
                "website": apollo_data.get('website_url'),
                "industry": apollo_data.get('industry'),
                "employee_count": apollo_data.get('estimated_num_employees'),
                "revenue": apollo_data.get('estimated_annual_revenue'),
                "founded_year": apollo_data.get('founded_year'),
                "linkedin_url": apollo_data.get('linkedin_url'),
                "phone": phone,
                "location": location_data,  # Enhanced location data
                "technologies": apollo_data.get('technologies', []),
                "funding": {
                    "total_funding": apollo_data.get('total_funding'),
                    "latest_funding_round": apollo_data.get('latest_funding_round'),
                    "latest_funding_amount": apollo_data.get('latest_funding_amount')
                }
            }
        except Exception as e:
            logger.error(f"Error formatting Apollo data: {e}")
            # Return a minimal set of data if we can
            return {
                "name": apollo_data.get('name', 'Unknown'),
                "website": apollo_data.get('website_url', 'Unknown'),
                "error": "Data formatting error"
            }
    
    def _format_address(self, apollo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format address data from Apollo.
        
        Args:
            apollo_data: The raw Apollo organization data
            
        Returns:
            dict: Formatted address data
        """
        # This is kept for backward compatibility
        return {
            "street": apollo_data.get('street_address'),
            "city": apollo_data.get('city'),
            "state": apollo_data.get('state'),
            "country": apollo_data.get('country'),
            "zip": apollo_data.get('postal_code')
        }

    def search_person(self, email: str) -> Optional[Dict[str, Any]]:
        """
        Search for a person by email.
        
        Args:
            email: The email to search for
            
        Returns:
            dict: The person data or None if not found
        """
        if not self.api_key:
            logger.warning(f"Cannot search for {email}: No Apollo API key")
            return None
            
        try:
            # Apollo's people/search endpoint
            endpoint = f"{self.base_url}/people/search"
            
            # Use the X-Api-Key header as required by Apollo
            headers = {
                "Content-Type": "application/json",
                "Cache-Control": "no-cache",
                "X-Api-Key": self.api_key
            }
            
            payload = {
                "q_person_email": email,
                "page": 1,
                "per_page": 1
            }
            
            logger.info(f"Searching for person with email: {email}")
            
            response = requests.post(endpoint, json=payload, headers=headers, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('people') and len(data['people']) > 0:
                    logger.info(f"Found person data for email {email}")
                    return self._format_person_data(data['people'][0])
                else:
                    logger.warning(f"No person found for email {email}")
            else:
                # Enhanced error logging
                try:
                    error_details = response.json()
                    logger.error(f"Apollo API error ({response.status_code}) for email search {email}: {error_details}")
                except:
                    logger.error(f"Apollo API error ({response.status_code}) for email search {email}: {response.text[:500]}")
                
            return None
            
        except Exception as e:
            logger.error(f"Error searching for person with Apollo: {e}")
            return None
            
    def _format_person_data(self, person_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format person data from Apollo.
        
        Args:
            person_data: The raw Apollo person data
            
        Returns:
            dict: Formatted person data
        """
        # Create location data if available
        location = None
        if person_data.get('city') or person_data.get('state') or person_data.get('country'):
            location = {
                "city": person_data.get('city'),
                "state": person_data.get('state'),
                "country": person_data.get('country'),
            }
            
            # Create formatted location string
            location_parts = []
            if location["city"]:
                location_parts.append(location["city"])
            if location["state"]:
                location_parts.append(location["state"])
            if location["country"]:
                location_parts.append(location["country"])
                
            if location_parts:
                location["formatted_address"] = ", ".join(location_parts)
        
        return {
            "name": f"{person_data.get('first_name', '')} {person_data.get('last_name', '')}".strip(),
            "first_name": person_data.get('first_name'),
            "last_name": person_data.get('last_name'),
            "title": person_data.get('title'),
            "seniority": person_data.get('seniority'),
            "email": person_data.get('email'),
            "linkedin_url": person_data.get('linkedin_url'),
            "phone": person_data.get('phone_number'),
            "department": person_data.get('department'),
            "company": person_data.get('organization_name'),
            "location": location
        }
