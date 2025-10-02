import os
import uuid

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

memory = InMemorySaver()
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "gemini-embedding-001"
BUS_STOPS_URL = ("https://pirdop.wordpress.com/%D1%80%D0%B0%D0%B7%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5"
                 "-%D0%BD%D0%B0-%D0%B0%D0%B2%D1%82%D0%BE%D0%B1%D1%83%D1%81%D0%B8/")

def get_web_docs(url):
    loader = WebBaseLoader(url)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    return split_docs

def define_model():
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer to user questions and using this context if it is needed: {context}"),
            ("human", "{input}")
        ]
    )

    return llm, prompt

def create_chain(llm, prompt):
    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    return chain

def create_vector_db(docs):
    embedding = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    vector_store = Chroma.from_documents(
        embedding=embedding,
        documents=docs
    )

    return vector_store

def create_retriever(llm, prompt, chain, vector_store):
    retriever = vector_store.as_retriever()
    history_retriever = create_history_aware_retriever(
        llm=llm,
        prompt=prompt,
        retriever=retriever
    )

    retrieval_chain: Runnable = create_retrieval_chain(
        retriever=history_retriever,
        combine_docs_chain=chain
    )

    return retrieval_chain

def define_tool_agent(retrieval_chain, llm):
    def run_retrieval(query: str | dict) -> str:
        print("Debug retriever runner: ", query)
        if isinstance(query, dict):
            agent_input = query.get("__arg1") or query.get("input") or str(query)
        else:
            agent_input = str(query)

        result = retrieval_chain.invoke({"input": agent_input})
        return result["answer"]

    retrieval_tool = Tool(
        name="DocTool",
        func=run_retrieval,
        description="Use this tool to answer questions about "
                    "bus and train schedules, stops, and times from the uploaded documents."
    )

    model_agent = create_react_agent(
        model=llm,
        tools=[retrieval_tool],
        checkpointer=memory
    )

    return model_agent

if __name__ == "__main__":
    urls = get_web_docs(BUS_STOPS_URL)
    model, model_prompt = define_model()
    model_chain = create_chain(model, model_prompt)
    vector_db = create_vector_db(urls)
    model_retriever = create_retriever(model, model_prompt, model_chain, vector_db)
    agent = define_tool_agent(model_retriever, model)
    thread_id = str(uuid.uuid4())

    while True:
        user_input = input("Your message: ")

        if user_input == "exit":
            break

        # This ensures model memory in a sertan conversation
        config = {"configurable": {"thread_id": thread_id}}

        # By documentation the agent has to be runed like this. Ignore the warning
        response = agent.invoke(
            {
                "messages": [{"role": "user", "content": user_input}]
            }, config
        )

        # Getting only the last answer from the agent
        messages = response.get("messages", [])

        last_ai_message = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None
        )

        if last_ai_message:
            print("Answer:", last_ai_message.content)
        else:
            print("No assistant response found")
