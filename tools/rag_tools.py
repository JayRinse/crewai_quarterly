from vectordb_processing.utils import (
	connection_string_create,
	create_db,
	documents_in_folder,
	custom_document_loader,
	create_nodes,
	vector_store_create,
	populated_tables,
	index_load,
	query_engine_create
)
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import PromptTemplate
from pprint import pprint
import json
from tools.llm_select import gemini_flash
from langchain.tools import tool

def table_select(naming: str=None):
	# naming = "questions", "dd"
	db_setup = {
		"llm": {
			"model": "models/gemini-1.5-flash-latest",
			"host": "cloud",
			"input_price": 0.35,
			"output_price": 1.05,
		},
		"vdb": {
			"db_name": f"embed001_{naming}_doc",
			"embedding": "models/embedding-001",
			"embed_dim": 768,
			"context_window": 8192,
			"embed_length": None,
			"chunk_size": 512,
			"chunk_overlap": 128,
			"protocol": "postgres",
			"hostname": "localhost",
			"username": "test1",
			"password": "test1",
			"port": 5432,
			"database": "postgres",
			"table_name": naming,
		},
	}
	return db_setup

class RAGTool():
	# def __init__(self, default_table_selection="questions"):
	#     self.default_table_selection = default_table_selection
	@tool("RAG_Tool")
	def search_rag(
		# self,
		# gen_input: dict=None,
		query_string: str=None,
		table_selection: str=None,
	):
		"""
		Useful tool to search through the due dilligence document, using table_selection = 'dd'.
		The input required is a query_string of the question to ask the document, and the table_selection of the document to query.
		"""
		# table_selection = task_gen_questions.get("table_selection", "questions")
		# table_selection = table_selection or self.default_table_selection

		# query_string = gen_input["query_string"]
		# table_selection = gen_input["table_selection"]
		# table_selection = "questions"

		llm, embed_model = gemini_flash(api_path="/home/johan/Code/secrets/google_api.json")

		db_setup = table_select(table_selection)
		
		connection_rag = connection_string_create(
		protocol=db_setup["vdb"]["protocol"],
		hostname=db_setup["vdb"]["hostname"],
		username=db_setup["vdb"]["username"],
		password=db_setup["vdb"]["password"],
		port=db_setup["vdb"]["port"],
		database=db_setup["vdb"]["db_name"],
		)
		
		vector_store = PGVectorStore.from_params(
		database=db_setup["vdb"]["db_name"],
		host=db_setup["vdb"]["hostname"],
		password=db_setup["vdb"]["password"],
		port=db_setup["vdb"]["port"],
		user=db_setup["vdb"]["username"],
		table_name=db_setup["vdb"]["table_name"],
		embed_dim=db_setup["vdb"]["embed_dim"],
		hybrid_search=True,
		text_search_config="english",
		)

		sotrage_context, hybrid_index = index_load(
		vdb_connection=connection_rag, vector_store=vector_store
		)
		
		query_engine = hybrid_index.as_query_engine(
		vector_store_query_mode="hybrid",
		similarity_top_k=7,
		vector_store_kwargs={
			"ivfflat_probes": 10,
			"hnsw_ef_search": 300
		},
		node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.40)],
		)

		prompt_tmpl = PromptTemplate("""\
		You are a professional investment analyst assisting a financial advisor who is doing a due dilligence. \n
		---------------------\n
		{context_str}
		---------------------\n
		Given the context information and not prior knowledge, answer the query.\n
		Please write the answer as an investment professional who is writing a formal reponse to a question, being as detailed as possible,
		explaining the thinking behind the answer. Do not reference any graphs.\n
		Query: {query_str}
		Answer: \
		""")

		query_engine.update_prompts(
		{"response_synthesizer:text_qa_template": prompt_tmpl}
		)
		prompts_dict = query_engine.get_prompts()

		query_embedding = embed_model.get_query_embedding(query_string)
		retrieval_response = query_engine.retrieve(
		query_string,
		)

		response = query_engine.query(
			query_string,
		)

		return response

	# @tool("RAG_Tool_Q")
	# def search_rag_q(
	# 	# self,
	# 	# gen_input: dict=None,
	# 	query_string: str=None,
	# 	table_selection: str="questions",
	# ):
	# 	"""
	# 	Useful tool to search through previous questions that have been used to query a due diligence document.
	# 	The input required is a query_string that describes what you're looking for.
	# 	"""
	# 	# table_selection = task_gen_questions.get("table_selection", "questions")
	# 	# table_selection = table_selection or self.default_table_selection

	# 	# query_string = gen_input["query_string"]
	# 	# table_selection = gen_input["table_selection"]
	# 	# table_selection = "questions"

	# 	llm, embed_model = gemini_flash(api_path="/home/johan/Code/secrets/google_api.json")

	# 	db_setup = table_select("questions")
		
	# 	connection_rag = connection_string_create(
	# 	protocol=db_setup["vdb"]["protocol"],
	# 	hostname=db_setup["vdb"]["hostname"],
	# 	username=db_setup["vdb"]["username"],
	# 	password=db_setup["vdb"]["password"],
	# 	port=db_setup["vdb"]["port"],
	# 	database=db_setup["vdb"]["db_name"],
	# 	)
		
	# 	vector_store = PGVectorStore.from_params(
	# 	database=db_setup["vdb"]["db_name"],
	# 	host=db_setup["vdb"]["hostname"],
	# 	password=db_setup["vdb"]["password"],
	# 	port=db_setup["vdb"]["port"],
	# 	user=db_setup["vdb"]["username"],
	# 	table_name=db_setup["vdb"]["table_name"],
	# 	embed_dim=db_setup["vdb"]["embed_dim"],
	# 	hybrid_search=True,
	# 	text_search_config="english",
	# 	)

	# 	sotrage_context, hybrid_index = index_load(
	# 	vdb_connection=connection_rag, vector_store=vector_store
	# 	)
		
	# 	query_engine = hybrid_index.as_query_engine(
	# 	vector_store_query_mode="hybrid",
	# 	similarity_top_k=7,
	# 	vector_store_kwargs={
	# 		"ivfflat_probes": 10,
	# 		"hnsw_ef_search": 300
	# 	},
	# 	node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.40)],
	# 	)

	# 	prompt_tmpl = PromptTemplate("""\
	# 	You are a professional investment analyst assisting a financial advisor who is doing a due dilligence. \n
	# 	---------------------\n
	# 	{context_str}
	# 	---------------------\n
	# 	Given the context information and not prior knowledge, answer the query.\n
	# 	Please write the answer as an investment professional who is writing a formal reponse to a question, being as detailed as possible,
	# 	explaining the thinking behind the answer. Do not reference any graphs.\n
	# 	Query: {query_str}
	# 	Answer: \
	# 	""")

	# 	query_engine.update_prompts(
	# 	{"response_synthesizer:text_qa_template": prompt_tmpl}
	# 	)
	# 	prompts_dict = query_engine.get_prompts()

	# 	query_embedding = embed_model.get_query_embedding(query_string)
	# 	retrieval_response = query_engine.retrieve(
	# 	query_string,
	# 	)

	# 	response = query_engine.query(
	# 		query_string,
	# 	)

	# 	return response

	# @tool("RAG_Tool_Ans")
	# def search_rag_dd(
	# 	# self,
	# 	# gen_input: dict=None,
	# 	query_string: str=None,
	# 	table_selection: str="dd",
	# 	):
	# 	"""
	# 	Useful tool to search through previous questions that have been used to query a due diligence document.
	# 	The input required is a query_string that describes what you're looking for.
	# 	"""
	# 	# table_selection = task_gen_questions.get("table_selection", "questions")
	# 	# table_selection = table_selection or self.default_table_selection

	# 	# query_string = gen_input["query_string"]
	# 	# table_selection = gen_input["table_selection"]
	# 	# table_selection = "questions"

	# 	llm, embed_model = gemini_flash(api_path="/home/johan/Code/secrets/google_api.json")

	# 	db_setup = table_select("dd")
		
	# 	connection_rag = connection_string_create(
	# 	protocol=db_setup["vdb"]["protocol"],
	# 	hostname=db_setup["vdb"]["hostname"],
	# 	username=db_setup["vdb"]["username"],
	# 	password=db_setup["vdb"]["password"],
	# 	port=db_setup["vdb"]["port"],
	# 	database=db_setup["vdb"]["db_name"],
	# 	)
		
	# 	vector_store = PGVectorStore.from_params(
	# 	database=db_setup["vdb"]["db_name"],
	# 	host=db_setup["vdb"]["hostname"],
	# 	password=db_setup["vdb"]["password"],
	# 	port=db_setup["vdb"]["port"],
	# 	user=db_setup["vdb"]["username"],
	# 	table_name=db_setup["vdb"]["table_name"],
	# 	embed_dim=db_setup["vdb"]["embed_dim"],
	# 	hybrid_search=True,
	# 	text_search_config="english",
	# 	)

	# 	sotrage_context, hybrid_index = index_load(
	# 	vdb_connection=connection_rag, vector_store=vector_store
	# 	)
		
	# 	query_engine = hybrid_index.as_query_engine(
	# 	vector_store_query_mode="hybrid",
	# 	similarity_top_k=7,
	# 	vector_store_kwargs={
	# 		"ivfflat_probes": 10,
	# 		"hnsw_ef_search": 300
	# 	},
	# 	node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.40)],
	# 	)

	# 	prompt_tmpl = PromptTemplate("""\
	# 	You are a professional investment analyst assisting a financial advisor who is doing a due dilligence. \n
	# 	---------------------\n
	# 	{context_str}
	# 	---------------------\n
	# 	Given the context information and not prior knowledge, answer the query.\n
	# 	Please write the answer as an investment professional who is writing a formal reponse to a question, being as detailed as possible,
	# 	explaining the thinking behind the answer. Do not reference any graphs.\n
	# 	Query: {query_str}
	# 	Answer: \
	# 	""")

	# 	query_engine.update_prompts(
	# 	{"response_synthesizer:text_qa_template": prompt_tmpl}
	# 	)
	# 	prompts_dict = query_engine.get_prompts()

	# 	query_embedding = embed_model.get_query_embedding(query_string)
	# 	retrieval_response = query_engine.retrieve(
	# 	query_string,
	# 	)

	# 	response = query_engine.query(
	# 		query_string,
	# 	)

	# 	return response






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