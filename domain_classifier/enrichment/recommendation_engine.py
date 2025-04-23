"""Recommendation engine for Domotz products based on company profile."""
import logging
from typing import Dict, Any, List

# Set up logging
logger = logging.getLogger(__name__)

class DomotzRecommendationEngine:
    def generate_recommendations(self, company_type: str, 
                               apollo_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate tailored Domotz recommendations based on company type and Apollo data.
        
        Args:
            company_type: The classified company type
            apollo_data: Enriched company data from Apollo
            
        Returns:
            dict: Tailored recommendations
        """
        employee_count = apollo_data.get('employee_count', 0) if apollo_data else 0
        technologies = apollo_data.get('technologies', []) if apollo_data else []
        
        # Base recommendations by company type
        if company_type == "Managed Service Provider":
            return self._generate_msp_recommendations(employee_count, technologies)
        elif company_type == "Integrator - Commercial A/V":
            return self._generate_commercial_av_recommendations(employee_count, technologies)
        elif company_type == "Integrator - Residential A/V":
            return self._generate_residential_av_recommendations(employee_count, technologies)
        elif company_type == "Internal IT Department":
            return self._generate_internal_it_recommendations(employee_count, technologies)
        else:
            return self._generate_generic_recommendations()
    
    def _generate_msp_recommendations(self, employee_count: int, 
                                    technologies: List[str]) -> Dict[str, Any]:
        """Generate MSP-specific recommendations."""
        use_cases = [
            "Multi-tenant monitoring for client networks",
            "Automated alerting for client infrastructure issues",
            "Remote troubleshooting without site visits"
        ]
        
        # Add specific recommendations based on size
        if employee_count > 50:
            use_cases.append("Enterprise-grade RMM integration")
            use_cases.append("White-label network monitoring for clients")
        
        # Add tech-specific recommendations
        if any(tech in technologies for tech in ["Cisco", "Meraki", "Ubiquiti"]):
            use_cases.append("Network device monitoring and configuration backups")
        
        return {
            "primary_value": "Reduce truck rolls while improving client satisfaction",
            "use_cases": use_cases,
            "recommended_plan": "MSP Plan" if employee_count > 20 else "Standard Plan"
        }
    
    def _generate_commercial_av_recommendations(self, employee_count: int, 
                                             technologies: List[str]) -> Dict[str, Any]:
        """Generate Commercial A/V-specific recommendations."""
        use_cases = [
            "Monitor complex A/V installations remotely",
            "Proactive alerts for failing commercial display systems",
            "Network health monitoring for conference room systems"
        ]
        
        if employee_count > 30:
            use_cases.append("Multi-site A/V system monitoring")
        
        return {
            "primary_value": "Ensure commercial A/V systems are always operational",
            "use_cases": use_cases,
            "recommended_plan": "Standard Plan"
        }
    
    def _generate_residential_av_recommendations(self, employee_count: int, 
                                              technologies: List[str]) -> Dict[str, Any]:
        """Generate Residential A/V-specific recommendations."""
        use_cases = [
            "Monitor home automation systems remotely",
            "Reduce service calls with proactive system monitoring",
            "Detect network issues affecting smart home functionality"
        ]
        
        return {
            "primary_value": "Deliver premium support without extra truck rolls",
            "use_cases": use_cases,
            "recommended_plan": "Standard Plan"
        }
    
    def _generate_internal_it_recommendations(self, employee_count: int, 
                                           technologies: List[str]) -> Dict[str, Any]:
        """Generate Internal IT-specific recommendations."""
        use_cases = [
            "Monitor critical network infrastructure",
            "Automate device inventory tracking",
            "Detect and alert on network anomalies"
        ]
        
        if employee_count > 100:
            use_cases.append("Enterprise network topology mapping")
            use_cases.append("Multi-site monitoring and management")
        
        return {
            "primary_value": "Increase IT efficiency with comprehensive network visibility",
            "use_cases": use_cases,
            "recommended_plan": "Enterprise Plan" if employee_count > 200 else "Standard Plan"
        }
    
    def _generate_generic_recommendations(self) -> Dict[str, Any]:
        """Generate generic recommendations."""
        return {
            "primary_value": "Gain complete visibility into your network",
            "use_cases": [
                "Monitor all network devices in real-time",
                "Detect and troubleshoot issues quickly",
                "Maintain accurate device inventory"
            ],
            "recommended_plan": "Standard Plan"
        }
