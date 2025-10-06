import datetime
import os
import uuid
from datetime import timezone
from typing import Any

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import Tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import create_react_agent
from psycopg_pool import ConnectionPool
from sqlalchemy import Column, String, Text, JSON, TIMESTAMP, create_engine
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

PG_USER: str = os.getenv("PG_USER")
PG_PASSWORD: str = os.getenv("PG_PASS")
PG_HOST: str = os.getenv("PG_HOST")
PG_PORT: str = os.getenv("PG_PORT")
DB_NAME: str = os.getenv("DB_NAME")

LLM_MODEL: str = "gemini-2.5-flash"
EMBEDDING_MODEL: str = "gemini-embedding-001"
DB_URI: str = f"postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{DB_NAME}"
BUS_STOPS_URL: str = ("https://pirdop.wordpress.com/%D1%80%D0%B0%D0%B7%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5"
                      "-%D0%BD%D0%B0-%D0%B0%D0%B2%D1%82%D0%BE%D0%B1%D1%83%D1%81%D0%B8/")

Base = declarative_base()


class Documents(Base):
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_type = Column(String, nullable=False)  # 'pdf', 'url', 'note', etc.
    source = Column(String, nullable=False)  # path or URL
    content = Column(Text, nullable=False)
    meta = Column(JSON, nullable=True)
    created_at = Column(TIMESTAMP, default=datetime.datetime.now(timezone.utc))


def create_database():
    pool = ConnectionPool(
        conninfo=DB_URI,
        max_size=20,
        kwargs={"autocommit": True}
    )

    # The pool provides required information. Ignore this warning
    checkpointer = PostgresSaver(pool)

    return checkpointer, pool


def get_web_docs(url):
    loader: WebBaseLoader = WebBaseLoader(url)
    documents: list[Document] = loader.load()

    return documents


def save_docs(documents: list[Document]):
    engine = create_engine(DB_URI)
    Base.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine)
    session = session_maker()

    for d in documents:
        text = session.query(Documents).filter(Documents.source == d.metadata["source"]).first()
        if not text:
            doc = Documents(source_type="url", source=d.metadata["source"], content=d.page_content)
            session.add(doc)

    session.commit()


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


def create_chain(llm: ChatGoogleGenerativeAI, prompt: ChatPromptTemplate):
    chain: Runnable = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

    return chain


def create_vector_db(documents):
    embedding = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    chunk_ids = []
    chunked_docs = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=250
    )

    for document in documents:
        split_docs: list[Document] = splitter.split_documents([document])
        for i, chunk in enumerate(split_docs):
            chunk_id = f"{document.id}-{i}"
            chunk_ids.append(chunk_id)
            chunked_docs.append(chunk)

    vector_store = Chroma.from_documents(
        embedding=embedding,
        documents=chunked_docs,
        persist_directory=os.getenv("CHROMA_PATH"),
        ids=chunk_ids
    )

    return vector_store


def create_retriever(llm: ChatGoogleGenerativeAI, prompt: ChatPromptTemplate, chain: Runnable, vector_store: Chroma):
    retriever: VectorStoreRetriever = vector_store.as_retriever()
    history_retriever: Runnable = create_history_aware_retriever(
        llm=llm,
        prompt=prompt,
        retriever=retriever
    )

    retrieval_chain: Runnable = create_retrieval_chain(
        retriever=history_retriever,
        combine_docs_chain=chain
    )

    return retrieval_chain


def define_tool_agent(retrieval_chain: Runnable, llm: ChatGoogleGenerativeAI):
    memory: PostgresSaver
    pool: ConnectionPool

    memory, pool = create_database()
    memory.setup()

    # Here the retriever is runed in a function so we can pass it to the Tool func kwarg
    def run_retrieval(query: str | dict) -> str:
        # Checking if model is using the tool, when giving answer
        print("Debug retriever runner: ", query)
        if isinstance(query, dict):
            agent_input = query.get("__arg1") or query.get("input") or str(query)
        else:
            agent_input = str(query)

        result: Any = retrieval_chain.invoke({"input": agent_input})
        return result["answer"]

    retrieval_tool = Tool(
        name="DocTool",
        func=run_retrieval,
        description="Use this tool to answer questions about "
                    "bus and train schedules, stops, and times from the uploaded documents."
    )

    agent: CompiledStateGraph = create_react_agent(
        model=llm,
        tools=[retrieval_tool],
        checkpointer=memory
    )

    return agent, pool


if __name__ == "__main__":
    model: ChatGoogleGenerativeAI
    model_prompt: ChatPromptTemplate
    model_agent: CompiledStateGraph
    model_pool: ConnectionPool

    docs = get_web_docs(BUS_STOPS_URL)
    model, model_prompt = define_model()
    model_chain: Runnable = create_chain(model, model_prompt)
    vector_db: Chroma = create_vector_db(docs)
    model_retriever: Runnable = create_retriever(model, model_prompt, model_chain, vector_db)
    model_agent, model_pool = define_tool_agent(model_retriever, model)
    thread_id: str = str(uuid.uuid4())
    previous_chat: str | None = input("Enter thread id, if you want new chat, press Enter: ")

    save_docs(docs)

    while True:
        user_input: str = input("Your message: ")

        if user_input == "exit":
            model_pool.close()
            break

        # This ensures model memory in a sertan conversation
        config: dict[str, dict] = {"configurable": {"thread_id": previous_chat if previous_chat else thread_id}}

        # By documentation the agent has to be runed like this. Ignore this warning
        response: dict[str, list] = model_agent.invoke(
            {
                "messages": [{"role": "user", "content": user_input}]
            }, config
        )

        # Getting only the answer from the agent
        messages: list = response.get("messages", [])

        last_ai_message: AIMessage = next(
            (msg for msg in reversed(messages) if isinstance(msg, AIMessage)),
            None
        )

        if last_ai_message:
            print("Answer:", last_ai_message.content)
        else:
            print("No assistant response found")
