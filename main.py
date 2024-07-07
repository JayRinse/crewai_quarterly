import os
from crewai import Agent, Task, Crew, Process
from textwrap import dedent
from tools.scraper_tools import ScraperTool
from tools.rag_tools import RAGTool

scrape_tool = ScraperTool().scrape
rag_tool = RAGTool().retrieve
class NewsletterCrew:
  def __init__(self, urls):
    self.urls = urls

  def run(self):
    scraper = Agent(
      role='Summarizer of Websites',
      goal='Ask the user for a list of URLs, then use the WebsiteSearchTool to then scrape the content, and provide the full content to the writer agent so it can then be summarized',
      backstory="""You work at a leading tech think tank.
      Your expertise is taking URLs and getting just the text-based content of them.""",
      verbose=True,
      allow_delegation=False,
      tools=[scrape_tool]
    )
    writer = Agent(
      role='Tech Content Summarizer and Writer',
      goal='Craft compelling short-form content on AI advancements based on long-form text passed to you',
      backstory="""You are a renowned Content Creator, known for your insightful and engaging articles.
      You transform complex concepts into compelling narratives.""",
      verbose=True,
      allow_delegation=True,
    )

    # Create tasks for your agents
    task1 = Task(
      description=f"""Take a list of websites that contain AI content, read/scrape the content and then pass it to the writer agent
      
      here are the URLs from the user that you need to scrape: {self.urls}""",
      agent=scraper
    )

    task2 = Task(
      description="""Using the text provided by the scraper agent, develop a short and compelling/interesting short-form summary of the 
      text provided to you about AI""",
      agent=writer
    )

    # Instantiate your crew with a sequential process
    NewsletterCrew = Crew(
      agents=[scraper, writer],
      tasks=[task1, task2],
      verbose=2, # You can set it to 1 or 2 to different logging levels
    )

    NewsletterCrew.kickoff()

if __name__ == "__main__":
  print("## Welcome to Newsletter Writer")
  print('-------------------------------')
  urls = input(
    dedent("""
      What is the URL you want to summarize?
    """))
  
  newsletter_crew = NewsletterCrew(urls)
  result = newsletter_crew.run()
  print("\n\n########################")
  print("## Here is the Result")
  print("########################\n")
  print(result)
# Define your agents with roles and goals

class QuarterlyLetterCrew:
  def __init__(self, topics):
    self.topics = topics

  def run(self):
    researcher = Agent(
      role='Retrieve information relevant to the topic provided by the user',
      goal='Ask the user for a list of topics, then use the RAG tool go and find relevant information on the topics, and provide the full content to the writer agent so it can be used to write a quarterly letter',
      backstory="""You work at a large financial advisory firm.
      Your expertise is taking topics and getting the relevant information using a RAG to provide to a financial writer.""",
      verbose=True,
      allow_delegation=False,
      tools=[rag_tool]
    )
    writer = Agent(
      role='Quarterly investment newsletter writer',
      goal='Craft compelling long-form detailed content on investment events over the previous quarter based on the text passed to you',
      backstory="""You are a renowned investment writer, known for your insightful and well structured quarterly investment reports.
      You transform complex concepts into compelling narratives.""",
      verbose=True,
      allow_delegation=True,
    )

    # Create tasks for your agents
    task1 = Task(
      description=f"""Take a list of topics that are relevant to the events over the previous quarter, read the content and then pass it to the writer agent
      
      Here are the topics from the user that you need to review: {self.topics}""",
      agent=researcher
    )

    task2 = Task(
      description="""Using the text provided by the researcher agent, develop a well structured quarterly report based on the provided text, ensuring that each topic is a paragraph long""",
      agent=writer
    )

    # Instantiate your crew with a sequential process
    QuarterlyLetterCrew = Crew(
      agents=[researcher, writer],
      tasks=[task1, task2],
      verbose=2, # You can set it to 1 or 2 to different logging levels
    )

    QuarterlyLetterCrew.kickoff()

if __name__ == "__main__":
  print("## Welcome to Quarterly Report Writer")
  print('-------------------------------')
  topics = input(
    dedent("""
      What is the topic you want to add to the report?
    """))
  
  report_crew = QuarterlyLetterCrew(topics)
  result = report_crew.run()
  print("\n\n########################")
  print("## Here is the Result")
  print("########################\n")
  print(result)