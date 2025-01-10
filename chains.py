
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_models import BedrockChat

from langchain_community.graphs import Neo4jGraph

from langchain_community.vectorstores import Neo4jVector

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

from typing import List, Any
from utils import BaseLogger, extract_region_and_data
from langchain_google_genai import GoogleGenerativeAIEmbeddings


def load_embedding_model(embedding_model_name: str, logger=BaseLogger(), config={}):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"], model="llama3.1:8b"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using OpenAI")
    elif embedding_model_name == "aws":
        embeddings = BedrockEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using AWS")
    elif embedding_model_name == "google-genai-embedding-001":        
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        dimension = 768
        logger.info("Embedding: Using Google Generative AI Embeddings")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
        )
        dimension = 384
        logger.info("Embedding: Using SentenceTransformer")
    return embeddings, dimension


def load_llm(llm_name: str, logger=BaseLogger(), config={}):
    if llm_name == "gpt-4":
        logger.info("LLM: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        logger.info("LLM: Using GPT-3.5")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif llm_name == "claudev2":
        logger.info("LLM: ClaudeV2")
        return BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 1024},
            streaming=True,
        )
    elif len(llm_name):
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            # seed=2,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=3072,  # Sets the size of the context window used to generate the next token.
        )
    logger.info("LLM: Using GPT-3.5")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps a support agent with answering programming questions.
    If you don't know the answer, just say that you don't know, you must not make up an answer. Answer only in Russian
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"question": user_input}, config={"callbacks": callbacks}
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_qa_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # RAG response
    general_system_template = """ 
    Use the following pieces of context to answer the question at the end.
    The context contains data entries and their associated regions.
    You should prefer information from the most relevant regions.
    Make sure to rely on information from the data entries to provide accurate responses.
    When you find particular data in the context useful, make sure to cite it in the answer using the link.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. Answer only in russian
    ----
    {summaries}
    ----
    Each answer you generate should contain a section at the end of links to 
    relevant regions and data entries you found useful, which are described under Source value.
    You can only use links to the regions that are present in the context and always
    add links to the end of the answer in the style of citations.
    Generate concise answers with a references sources section of links to 
    relevant regions only at the end of the answer. Answer only in russian
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    # Vector + Knowledge Graph response
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database="neo4j",  # neo4j by default
        index_name="region_data",  # renamed index
        text_node_property="description",  # renamed property
        retrieval_query="""
        WITH node AS data, score AS similarity
        CALL  { 
            WITH data
            MATCH (data)-[:LOCATED_IN]->(region)
            WITH region
            ORDER BY coalesce(region.relevance, 0) DESC
            WITH collect(region)[..2] as regions
            RETURN reduce(str='', region IN regions | str + 
                    '\n### Region: '+ coalesce(region.name, 'Unknown') + 
                    ' (Relevance: ' + coalesce(toString(region.relevance), '0') + '): ' + 
                    coalesce(region.description, 'No description available') + '\n') as regionTexts
        } 
        RETURN '##Data: ' + coalesce(data.title, 'No title available') + '\n' + 
            coalesce(data.description, 'No description available') + '\n' + 
            regionTexts AS text, similarity as score, {source: coalesce(data.link, 'No link available')} AS metadata
        ORDER BY similarity ASC
        """,
    )

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 2}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=3375,
    )
    return kg_qa


def generate_ticket(neo4j_graph, llm_chain, input_data):
    # Get high-ranked regions
    records = neo4j_graph.query(
        "MATCH (r:Region)-[:HAS_DATA]->(d:Data) RETURN r.name AS name, d.content AS content ORDER BY d.score DESC LIMIT 3"
    )
    regions = []
    for i, region in enumerate(records, start=1):
        regions.append((region["name"], region["content"]))
    # Ask LLM to generate new data in the same style
    regions_prompt = ""
    for i, region in enumerate(regions, start=1):
        regions_prompt += f"{i}. \n{region[0]}\n----\n\n"
        regions_prompt += f"{region[1][:150]}\n\n"
        regions_prompt += "----\n\n"

    gen_system_template = f"""
    You're an expert in generating high-quality content. 
    Formulate new data in the same style and tone as the following example regions and their data.
    {regions_prompt}
    ---

    Don't make anything up; only use information in the following input data.
    Return a name for the region and the data content itself.

    Return format template:
    ---
    Region: New region name
    Data: New data content
    ---
    """
    # we need jinja2 since the regions themselves contain curly braces
    system_prompt = SystemMessagePromptTemplate.from_template(
        gen_system_template, template_format="jinja2"
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            SystemMessagePromptTemplate.from_template(
                """
                Respond in the following template format or you will be unplugged.
                ---
                Region: New region name
                Data: New data content
                ---
                """
            ),
            HumanMessagePromptTemplate.from_template("{data}"),
        ]
    )
    llm_response = llm_chain(
        f"Here's the data to rewrite in the expected format: ```{input_data}```",
        [],
        chat_prompt,
    )
    new_region, new_data = extract_region_and_data(llm_response["answer"])
    return (new_region, new_data)

