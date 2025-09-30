import os

from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.agents import Tool, initialize_agent, AgentType, AgentExecutor
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

load_dotenv()

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

def create_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer to user questions and using this context if it is needed: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ]
    )

    chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )

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

def create_model(retrieval_chain, user_input, chat_history):
    response = retrieval_chain.invoke({
        "input": user_input,
        "chat_history": chat_history,
        "context": ""
    })

    return response


if __name__ == "__main__":
    urls = get_web_docs(BUS_STOPS_URL)
    vector_db = create_vector_db(urls)
    retriever = create_chain(vector_db)
    chat_history = []

    while True:
        user_input = input("Your message: ")

        if user_input == "exit":
            break

        response = create_model(retriever, user_input, chat_history)
        chat_history.append(HumanMessage(user_input))
        chat_history.append(AIMessage(response["answer"]))

        print("Answer:", response["answer"])
