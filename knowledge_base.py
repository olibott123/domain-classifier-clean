import csv
import requests
import time
import os
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Configuration
INPUT_CSV = "example_domains.csv"
OUTPUT_CSV = "knowledge_base.csv"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 2  # seconds

def extract_text_from_html(html_content):
    """Extract readable text from HTML content using BeautifulSoup"""
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove script and style elements
        for script_or_style in soup(['script', 'style', 'header', 'footer', 'nav']):
            script_or_style.extract()
            
        # Get text
        text = soup.get_text()
        
        # Clean up text: break into lines and remove leading/trailing space
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def fetch_domain_content(domain):
    """Fetch content for a domain"""
    # Make sure domain has a scheme
    if not domain.startswith('http'):
        url = f"https://{domain}"
    else:
        url = domain
    
    # Extract just the domain part for logging
    parsed_domain = urlparse(url).netloc or domain
    
    print(f"Fetching content for {parsed_domain}...")
    
    for attempt in range(MAX_RETRIES):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            
            if response.status_code == 200:
                content = extract_text_from_html(response.text)
                if content and len(content) > 100:
                    return content
                else:
                    print(f"  Warning: Content too short ({len(content) if content else 0} chars)")
            else:
                print(f"  Error: Status code {response.status_code}")
                
            # Try www subdomain if first attempt failed and no www already
            if attempt == 0 and not parsed_domain.startswith('www.'):
                url = f"https://www.{parsed_domain}"
                print(f"  Retrying with www: {url}")
                continue
                
        except requests.exceptions.RequestException as e:
            print(f"  Request error ({attempt+1}/{MAX_RETRIES}): {e}")
        
        time.sleep(DELAY_BETWEEN_REQUESTS)  # Wait before retrying
    
    print(f"  Failed to fetch content for {parsed_domain} after {MAX_RETRIES} attempts")
    return None

def build_knowledge_base():
    """Build a knowledge base CSV from example domains"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(OUTPUT_CSV) if os.path.dirname(OUTPUT_CSV) else '.', exist_ok=True)
    
    # Check if knowledge base already exists
    if os.path.exists(OUTPUT_CSV):
        print(f"{OUTPUT_CSV} already exists. Appending to it.")
        # Read existing domains to avoid duplicates
        existing_domains = set()
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if row and len(row) > 0:
                    existing_domains.add(row[0])
        print(f"Found {len(existing_domains)} existing domains in knowledge base")
    else:
        existing_domains = set()
        # Create knowledge base file with headers
        with open(OUTPUT_CSV, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['domain', 'company_type', 'content'])
    
    # Read example domains
    domains_to_process = []
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['domain'] not in existing_domains:
                domains_to_process.append(row)
    
    print(f"Processing {len(domains_to_process)} new domains...")
    
    # Process each domain
    success_count = 0
    for entry in domains_to_process:
        domain = entry['domain']
        company_type = entry['company_type']
        
        # Fetch content
        content = fetch_domain_content(domain)
        
        # Save to knowledge base
        with open(OUTPUT_CSV, 'a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            if content:
                writer.writerow([domain, company_type, content])
                print(f"✓ Successfully added {domain} ({len(content)} chars)")
                success_count += 1
            else:
                # Still save the domain with empty content for tracking
                writer.writerow([domain, company_type, ""])
                print(f"✗ Added {domain} with empty content")
        
        # Be nice to servers
        time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print(f"\nKnowledge base build complete:")
    print(f"- {success_count} of {len(domains_to_process)} domains successfully processed")
    print(f"- Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    build_knowledge_base()
