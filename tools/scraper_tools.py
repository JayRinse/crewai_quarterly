import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import re

class URLTool():
  @tool("URL Tool")
  def search_ddg(
      search_query: str=None
  ):
      "Useful tool to search on duckduckgoto find URLs which may contain valuvable information to scrape using the Scraper Tool."
      wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", max_results=1)
      search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")
      results = search.run(search_query)
      url_pattern = re.compile(r'http[^]]+')
      url_results = url_pattern.findall(results)
      return url_results

# class ScraperTool():
#   @tool("Scraper Tool")
#   def scrape(
#     url_results: list=None
#     ):
#     "Useful tool to scrape website content, use to learn more about a given url."

#     headers = {
#       'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

#     response = requests.get(url, headers=headers)
    
#     # Check if the request was successful
#     if response.status_code == 200:
#         # Parse the HTML content of the page
#         soup = BeautifulSoup(response.text, 'html.parser')

#         article = soup.find(id='insertArticle')
        
#         if article:
#             # Extract and print the text from the article
#             text = (article.get_text(separator=' ', strip=True))
#         else:
#             print("Article with specified ID not found.")
        
#         return text
#     else:
#         print("Failed to retrieve the webpage")
    

import requests
from bs4 import BeautifulSoup
from typing import List, Optional

class ScraperTool:
    @tool('Scraper Tool')
    def scrape(url_results: Optional[List[str]] = None) -> str:
        """
        A useful tool used to scrape websites given a list of URLs.
        """
        if not url_results:
            return "No URLs provided for scraping."

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
        }
        
        scraped_content = []

        for url in url_results:
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find all header tags and paragraphs
                elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
                
                if elements:
                    content = []
                    for element in elements:
                        tag = element.name
                        text = element.get_text(strip=True)
                        if text:  # Only add non-empty elements
                            content.append(f"<{tag}>{text}</{tag}>")
                    
                    scraped_content.append(f"Content from {url}:\n" + "\n".join(content) + "\n")
                else:
                    scraped_content.append(f"No relevant content found on {url}\n")
            except requests.RequestException as e:
                scraped_content.append(f"Failed to retrieve {url}: {str(e)}\n")

        return "\n".join(scraped_content)

# class ScraperTool:
#     @tool("Scraper Tool")
#     def scrape(url_results: list=None):
#         """
#         Scrapes content from multiple websites and concatenates the results.

#         Args:
#             url_results: A list of URLs to scrape.

#         Returns:
#             A string containing the concatenated scraped content.
#         """

#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
#         }
        
#         scraped_content = ""  # Initialize empty string to store all content

#         for url in url_results:
#             response = requests.get(url, headers=headers)

#             if response.status_code == 200:
#                 soup = BeautifulSoup(response.text, 'html.parser')
#                 article = soup.find(id='insertArticle')  # Assuming consistent article structure

#                 if article:
#                     text = article.get_text(separator=' ', strip=True)
#                     scraped_content += text + "\n\n"  # Add newline between articles
#                 else:
#                     print(f"Article with specified ID not found on {url}")
#             else:
#                 print(f"Failed to retrieve {url}")

#         return scraped_content