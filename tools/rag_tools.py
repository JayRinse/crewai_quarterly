from langchain.tools import tool

import psycopg2
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, PromptTemplate #, Settings, QueryBundle
from llama_index.core.postprocessor import SimilarityPostprocessor
from llm_select import gemini_flash

selection = 'geimini15flash'

llm, embed_model = gemini_flash()

def connection_string_create(
        protocol: str = "postgres",
        hostname: str = "localhost",
        username: str = None,
        password: str = None,
        port: int = 5432,
        database: str = "postgres"
        ):
    connection_string = f"{protocol}://{username}:{password}@{hostname}:{port}/{database}"
    return connection_string

model_dict = {'geimini15flash': {
        'llm': 'models/gemini-1.5-flash-latest',
        'embedding': 'models/embedding-001',
        'embed_dim': 768,
        'context_window': 8192,
        'embed_length': None,
        'chunk_size': 512,
        'chunk_overlap': 128,
        'host': 'cloud',
        'db_name': 'embed001_db',
        'input_price': 0.35,
        'output_price': 1.05
    }
}

db_name = model_dict[selection]['db_name']
print(f"{db_name} is being used")

vdb = {
    "protocol": "postgres",
    "hostname": "localhost",
    "username": "test1",
    "password": "test1",
    "port": 5432,
    "database": "postgres",
    "vdb": db_name
}

class RAGTool():

    @tool("RAG Tool")
    def retrieve(
        query_string: str=None,
    ):
        """
        A tool used to retrieve information from a database and synthesize it into an answer.
        """
        vector_store = PGVectorStore.from_params(
            database=db_name,
            host=vdb['hostname'],
            password=vdb['password'],
            port=vdb['port'],
            user=vdb['username'],
            table_name='idea_farm',
            embed_dim=model_dict[selection]['embed_dim'],
            hybrid_search=True,
            text_search_config="english"
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )
        hybrid_index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            show_progress=False,
        )

        prompt_tmpl = PromptTemplate("""\
        You are a professional investment analyst assisting a financial advisor. \n
        ---------------------\n
        {context_str}
        ---------------------\n
        Given the context information and not prior knowledge, answer the query.\n
        Please write the answer as an investment professional who is writing a structured article, being as detailed as possible,
        explaining the thinking behind the answer. Do not reference any graphs.\n
        Query: {query_str}
        Answer: \
        """)

        query_engine = hybrid_index.as_query_engine(
            vector_store_query_mode="hybrid",
            similarity_top_k=7,
            vector_store_kwargs={
                "ivfflat_probes": 10,
                "hnsw_ef_search": 300
            },
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.40)],
        )

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
    
# rag = RAGTool()
# response = rag.retrieve('tell me about industrials')
# print(response)