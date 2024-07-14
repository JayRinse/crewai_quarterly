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

rag_tool_q = RAGTool.search_rag_q
rag_tool_dd = RAGTool.search_rag_dd

class QuarterlyLetterCrew:
	def __init__(self):
		pass

	def run(self):
		q_generator = Agent(
			role="Question_Generator_Agent",
			goal="Identify relevant questions to be used to query a due dilligence document, to identify the strenghts and weaknesses of a fund",
			backstory="""
		You are an expert fund research analyst at a hedge fund, who is skilled in identifying meaningful questions
		which can be used to gain insight into the strengths and weaknesses of a fund.
		""",
			instructions="""
		When you need to search for information, use the rag_tool_q. 
		To use this tool, provide a query_string that describes what you're looking for. 
		Also provide the table_selection which is "questions".
		For example: {"query_string": "What are common questions for fund due diligence?",
		"table_selection": "questions"}
		""",
			verbose=True,
			allow_delegation=False,

			tools=[rag_tool_q],
			llm=langchain_llm,
			max_iter=1,
		)
		
		ans_generator = Agent(
			role="Answering_Agent",
			goal="Answer the questions provided by the Question_Generator_Agent.",
			backstory="""
		You are an expert researcher at a hedge fund, who is skilled in finding answers to questions provided to you by the Question_Generator_Agent.
		""",
			instructions="""
		Answer the questions given to you by the Question_Generator_Agent. You must answer those specific questions by using rag_tool_dd. Systematically
		answer the questions provided to you by the Question_Generator_Agent by using the questions and querying the due diligence document using the rag_tool_dd.
		Also provide the table_selection which is "dd".
		For example, if the Questions_Generator_Agent asked "What types of investments will the fund pursue?, you are able to use the rag_tool_dd
		to ask that exact question and record its reponse. For example: {"query_string": "What types of investments will the fund pursue?", "table_selection": "dd"}
		""",
			verbose=True,
			allow_delegation=False,

			tools=[rag_tool_dd],
			llm=langchain_llm,
			max_iter=5,
		)
		
		task_gen_questions = Task(
			description="""
		Clearly identify a list of 5 questions that can be used to interrogate a due dilligence document of a fund.
		""",
			# {self.topics}, {self.timeframe}
			agent=q_generator,
			expected_output="A list of 5 questions that can be used to interrogate a due dilligence document of a fund.",
			metadata={"table_selection": "questions"},
		)
		
		task_ans_questions = Task(
			description="""
		Try to find the answers to the questions posed by Question_Generator_Agent.
		""",
			agent=ans_generator,
			expected_output="A list of all the answers to the questions provided to you.",
			context=[task_gen_questions],
			metadata={"table_selection": "dd"},
		)

		QuarterlyLetterCrew = Crew(
			# agents=[researcher, writer, assistant],
			# tasks=[task1, task2, task3, task4],
			agents=[q_generator, ans_generator],
			tasks=[task_gen_questions, task_ans_questions],
			verbose=2,  # You can set it to 1 or 2 to different logging levels
			process=Process.sequential,
			# manager_llm=manager_llm
			# process=Process.hierarchical,
		)

		QuarterlyLetterCrew.kickoff()

		result = QuarterlyLetterCrew.kickoff()
		print(result)
		return result


if __name__ == "__main__":
	print("## Welcome to Quarterly Report Writer")
	print("-------------------------------")
	# topics = input(
	#   dedent("""
	#     What are the topics/questions you want to add to the report?
	#   """))
	# timeframe = input(
	#   dedent("""
	#     What are the quarter and year?
	#   """))

	report_crew = QuarterlyLetterCrew()
	result = report_crew.run()
	print("\n\n########################")
	print("## Here is the Result")
	print("########################\n")
	print(result)
	# with open('output.md', "w") as file:
	#   print('\n\nThese results have been exported to output.md')
	#   file.write(result)
	print(f"""
    Task completed!
    Task: {report_crew.task_gen_questions.output.description}
    Output: {report_crew.task_gen_questions.output.raw_output}
""")