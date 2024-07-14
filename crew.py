import os
from crewai import Agent, Task, Crew, Process
from textwrap import dedent

# from tools.scraper_tools import ScraperTool, URLTool
from tools.rag_tools import RAGTool
import vectordb_processing
from tools.llm_select import gemini_flash, langchain_gemini_flash

# from llama_index.llms.ollama import Ollama
# from langchain.llms import Ollama
from langchain_community.llms import Ollama
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

# rag_tool_q = RAGTool.search_rag_q
# rag_tool_dd = RAGTool.search_rag_dd
rag_tool = RAGTool.search_rag

question_generator = Agent(
	role="Question_Generator_Agent",
	goal="Interrogate a due dilligence document for key insights about the fund.",
	backstory="""
You are an expert fund research analyst at a hedge fund, who is skilled at gaining key insights from a due dilligence document.
""",
	instructions="""
Use the due dilligence document to ask questions and gain insights about the fund. Fund performance is not an important question. This can be done via the rag_tool using the following format:
To use this tool, provide a query_string that describes what you're looking for, using table_selection = "dd".
For example: "query_string" = "What is the funds iinvestment objective?", "table_selection" = "dd"
""",
	verbose=True,
	allow_delegation=False,

	tools=[rag_tool],
	llm=langchain_llm,
	max_iter=10,
)

writer = Agent(
	role="Writer_Agent",
	goal="Write an investment report highlighting key insights from the questions and answered provided by the Questions_Generator_Agent.",
	backstory="""
You are an expert report writer who is adept at taking questions and answers and drawing key insights from them and writing an investment report.
""",
	instructions="""
Use the questions and answers from the Questions_Generator_Agent to write an investment report.
""",
	verbose=True,
	allow_delegation=False,
	llm=langchain_llm,
	max_iter=10,
)



# ans_generator = Agent(
# 	role="Answering_Agent",
# 	goal="Answer the questions you have been given.",
# 	backstory="""
# You are an expert researcher at a hedge fund, who is skilled in finding answers to questions provided to you.
# """,
# 	instructions="""
# Answer the questions given to you. You can answer those specific questions by using rag_tool_ddif it is needed. Systematically
# answer these questions. You can query the due diligence document using the rag_tool_dd.
# Also provide the table_selection which is "dd".
# For example, if the Questions_Generator_Agent asked "What types of investments will the fund pursue?, you are able to use the rag_tool_dd
# to ask that exact question and record its reponse. For example: {"query_string": "What types of investments will the fund pursue?", "table_selection": "dd"}
# """,
# 	verbose=True,
# 	allow_delegation=False,

# 	tools=[rag_tool_dd],
# 	llm=langchain_llm,
# 	max_iter=5,
# )

task_genans_questions = Task(
	description="""
Clearly identify questions to ask about a given fund to figure out key insights into the fund.
""",
	# {self.topics}, {self.timeframe}
	agent=question_generator,
	expected_output="Questions which can be used to interrogate a due dilligence document for key insights.",
	# metadata={"table_selection": "questions"},
)

task_writer = Task(
	description="""
Write an feedback report highlighting key insights.
""",
	# {self.topics}, {self.timeframe}
	agent=writer,
	expected_output="A detailed report of key insights.",
    context=[task_genans_questions]
	# metadata={"table_selection": "questions"},
)

# task_ans_questions = Task(
# 	description="""
# Find the answers to the questions that were given to you.
# """,
# 	agent=ans_generator,
# 	expected_output="A list of all the answers to the questions provided to you.",
# 	context=[task_gen_questions],
# 	metadata={"table_selection": "dd"},
# )

QuarterlyLetterCrew = Crew(
	# agents=[researcher, writer, assistant],
	# tasks=[task1, task2, task3, task4],
	agents=[question_generator, writer],
	tasks=[task_genans_questions, task_writer],
	verbose=2,  # You can set it to 1 or 2 to different logging levels
	process=Process.sequential,
	# manager_llm=manager_llm
	# process=Process.hierarchical,
)

result = QuarterlyLetterCrew.kickoff()
# 	print(result)
# 	return result


# if __name__ == "__main__":
# 	print("## Welcome to Quarterly Report Writer")
# 	print("-------------------------------")
# 	# topics = input(
# 	#   dedent("""
# 	#     What are the topics/questions you want to add to the report?
# 	#   """))
# 	# timeframe = input(
# 	#   dedent("""
# 	#     What are the quarter and year?
# 	#   """))

# 	report_crew = QuarterlyLetterCrew()
# 	result = report_crew.run()
# 	print("\n\n########################")
# 	print("## Here is the Result")
# 	print("########################\n")
# 	print(result)
# 	# with open('output.md', "w") as file:
# 	#   print('\n\nThese results have been exported to output.md')
# 	#   file.write(result)
# 	print(f"""
#     Task completed!
#     Task: {report_crew.task_gen_questions.output.description}
#     Output: {report_crew.task_gen_questions.output.raw_output}
# """)