import os
from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from tools.scraper_tools import ScraperTool, URLTool
import vectordb_processing
from tools.llm_select import gemini_flash, langchain_gemini_flash
# from llama_index.llms.ollama import Ollama
from langchain.llms import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

### WORKS ###
# os.environ["OPENAI_API_KEY"] = "NA"

# OPENAI_API_BASE='http://localhost:11434'
# OPENAI_MODEL_NAME='gemma:2b'
# OPENAI_MODEL_NAME='llama3:8b'
# OPENAI_API_KEY=''

# llm = Ollama(
#     # model = "gemma:2b",
#     model='llama3:8b',
#     base_url = "http://localhost:11434")
###

langchain_llm = langchain_gemini_flash()
# manager_llm = langchain_gemini_flash()

search_tool = URLTool().search_ddg
scraper_tool = ScraperTool().scrape
# rag_tool = RAGTool().retrieve
vdb_tool = VDBTool().query

class QuarterlyLetterCrew:
  def __init__(self, topics, timeframe):
    self.topics = topics
    self.timeframe = timeframe

  def run(self):
    # researcher = Agent(
    #   role='Retrieve information from the database relevant to the topic provided by the user and do not edit it',
    #   goal='Ask the user for a list of topics/questions, then use the RAG tool to go and find relevant information on the topics/questions, and provide the full content to the writer agent so it can be used to write a quarterly letter',
    #   backstory="""You work at a large financial advisory firm.
    #   Your expertise lie in taking topics and getting the relevant information using a RAG to provide the information to a financial writer.""",
    #   verbose=True,
    #   allow_delegation=False,
    #   # tools=[rag_tool, vdb_tool],
    #   tools=[rag_tool],
    #   llm=langchain_llm
    # )
    url_researcher = Agent(
      role='Retrieve relevant URLs from the internet related to the topic and timeframe indicated by the user. Then pass these URLs on the the text researcher agent',
      goal='Ask the user for a list of topics/questions, then use the URL tool to find a maximum of 5 URLs to use for relevant information on the topics/questions, and provide the full lsit of URLs to the text researcher agent',
      backstory="""You work at a large financial advisory firm.
      Your expertise lie in taking topics and finding URLs related to those topics to assist the text_researcher to get information from websites.""",
      verbose=True,
      allow_delegation=False,
      # tools=[rag_tool, vdb_tool],
      tools=[search_tool],
      llm=langchain_llm,
      max_iter=1
    )
    text_researcher = Agent(
      role='Retrieve relevant information from URLs which are provided to you by the url_researcher agent. Then use the URLs to scrape the content and pass it to the writer agent',
      goal='Use the URL list provided by the url_researcher to scrape websites to use for relevant information on the topics/questions, and provide the full content to the writer agent so it can be used to write a quarterly letter',
      backstory="""You work at a large financial advisory firm.
      Your expertise lie in taking topics and getting the relevant information using URLs and scraping them to provide the information to a financial writer.""",
      verbose=True,
      allow_delegation=False,
      # tools=[rag_tool, vdb_tool],
      tools=[scraper_tool],
      llm=langchain_llm,
      max_iter=1
    )
    writer = Agent(
      role='Quarterly investment newsletter writer',
      goal='Write a compelling and detailed investment report focusing on events from the previous quarter based on the text passed to you by the researcher',
      backstory="""You are a renowned investment writer, known for your insightful and well structured quarterly investment reports.
      You transform complex concepts into compelling narratives for investment clients.""",
      verbose=True,
      allow_delegation=False,
      llm=langchain_llm,
      max_iter=3
    )

    # manager = Agent(
    #     role="Project Manager",
    #     goal="Efficiently manage the crew and ensure high-quality task completion. Ensure the agents only attempt to complete a task once.",
    #     backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
    #     allow_delegation=True,
    # )

    # assistant = Agent(
    # role='Research Assistant',
    # goal='Provide additional information to support the writer using the vector database',
    # backstory="""You are a diligent research assistant with access to a vast database of financial information. 
    # Your job is to supplement the writer's work with relevant data and insights.""",
    # verbose=True,
    # allow_delegation=False,
    # tools=[vdb_tool],
    # llm=langchain_llm
# )

    # task = Task(
    #   description="Generate a querterly report on the provided topics for the period specified. Find URLs related to the topics and timeframe, before scraping them and writing a structured report with an opening paragraph, body and closing paragraph. Return completed, edited report.",
    #   expected_output="A comprehensive quarterly investment report on the provided topics for the provided timeframe",
    # )
    # Create tasks for your agents
    task1 = Task(
      description=f"""Take a list of topics that are provided by the user and retrieve relevant URLs by getting the URLs using the search 
      tool and then passing the list to the test_researcher agent. You can ask 5 queries about the topic to get relevant URLs for the text_researcher.
      Here are the topics from the user that you need to review: {self.topics}, and the quarter and year: {self.timeframe}.""",
      agent=url_researcher,
      expected_output="Comprehensive list of URLs to be passed to the text_researcher.",
    )
    task2 = Task(
      description=f"""Take a list of URLs provided by the url_researcher and scrape the webistes for relevant information related to the 
      topics that are provided by the user. Once you have enough relevant information, pass all the information to the writer agent who will write a quarterly report
      on the information provided.""",
      agent=text_researcher,
      expected_output="Comprehensive information about the topics provided for the timeframe provided to be passed to the writer agent.",
    )
    # task1 = Task(
    #   description=f"""Take a list of topics that are provided by the user and retrieve relevant information from the nodes using the VDB Tool 
    #   and then pass it to the writer agent. Use multiple queries about the topic to get relevant content which will be usable by the writer.
    #   Only ask the the VDB Tool one question at a time, with a maximum of 5 different queries.
    #   Here are the topics from the user that you need to review: {self.topics}, and the quarter and year: {self.timeframe}.""",
    #   agent=researcher,
    #   expected_output="Comprehensive reesearch of the current topics, including key insights, relevant data, and future predictions."
    # )

    task3 = Task(
      description="""Using the text provided by the researcher agent, develop a well structured quarterly report based on the provided text, 
      ensuring that each topic is a single paragraph. It should also contain an opening paragraph and a closing paragraph.""",
      agent=writer,
      expected_output="A well-written, comprehensive quarterly report on the given topics, including key insights, relevant data, and future predictions with explanations."
    )

    # task3 = Task(
    #     description="""Review the writer's draft and use the vdb_tool to find additional relevant information. 
    #     Focus on supporting data, recent trends, or expert opinions that could enhance the report.""",
    #     agent=assistant,
    #     expected_output="Additional relevant information and insights to enhance the quarterly report."
    # )

    # task4 = Task(
    #     description="""Using the text provided by the researcher agent and the additional information from the assistant, 
    #     develop a final well-structured quarterly report. Ensure each topic is a single paragraph, with an opening and closing paragraph.""",
    #     agent=writer,
    #     expected_output="A comprehensive, well-written quarterly report incorporating all provided information and insights."
    # )
    
    # Instantiate your crew with a sequential process
    QuarterlyLetterCrew = Crew(
      # agents=[researcher, writer, assistant],
      # tasks=[task1, task2, task3, task4],
      agents=[url_researcher, text_researcher, writer],
      tasks=[task1, task2, task3],
      verbose=2, # You can set it to 1 or 2 to different logging levels
      # process=Process.hierarchical,
      # manager_llm=manager_llm
      # process=Process.hierarchical,
    )

    QuarterlyLetterCrew.kickoff()
      
    result = QuarterlyLetterCrew.kickoff()
    return result

if __name__ == "__main__":
  print("## Welcome to Quarterly Report Writer")
  print('-------------------------------')
  topics = input(
    dedent("""
      What are the topics/questions you want to add to the report?
    """))
  timeframe = input(
    dedent("""
      What are the quarter and year?
    """))
  
  report_crew = QuarterlyLetterCrew(topics, timeframe)
  result = report_crew.run()
  print("\n\n########################")
  print("## Here is the Result")
  print("########################\n")
  print(result)



# scrape_tool = ScraperTool().scrape

# class NewsletterCrew:
#   def __init__(self, urls):
#     self.urls = urls

#   def run(self):
#     scraper = Agent(
#       role='Summarizer of Websites',
#       goal='Ask the user for a list of URLs, then use the WebsiteSearchTool to then scrape the content, and provide the full content to the writer agent so it can then be summarized',
#       backstory="""You work at a leading tech think tank.
#       Your expertise is taking URLs and getting just the text-based content of them.""",
#       verbose=True,
#       allow_delegation=False,
#       tools=[scrape_tool]
#     )
#     writer = Agent(
#       role='Tech Content Summarizer and Writer',
#       goal='Craft compelling short-form content on AI advancements based on long-form text passed to you',
#       backstory="""You are a renowned Content Creator, known for your insightful and engaging articles.
#       You transform complex concepts into compelling narratives.""",
#       verbose=True,
#       allow_delegation=True,
#     )

#     # Create tasks for your agents
#     task1 = Task(
#       description=f"""Take a list of websites that contain AI content, read/scrape the content and then pass it to the writer agent
      
#       here are the URLs from the user that you need to scrape: {self.urls}""",
#       agent=scraper
#     )

#     task2 = Task(
#       description="""Using the text provided by the scraper agent, develop a short and compelling/interesting short-form summary of the 
#       text provided to you about AI""",
#       agent=writer
#     )

#     # Instantiate your crew with a sequential process
#     NewsletterCrew = Crew(
#       agents=[scraper, writer],
#       tasks=[task1, task2],
#       verbose=2, # You can set it to 1 or 2 to different logging levels
#     )

#     NewsletterCrew.kickoff()

# if __name__ == "__main__":
#   print("## Welcome to Newsletter Writer")
#   print('-------------------------------')
#   urls = input(
#     dedent("""
#       What is the topic you want to write about?
#     """))
  
#   newsletter_crew = NewsletterCrew(urls)
#   result = newsletter_crew.run()
#   print("\n\n########################")
#   print("## Here is the Result")
#   print("########################\n")
#   print(result)
# Define your agents with roles and goals