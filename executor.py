import uuid

from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.state import CompiledStateGraph
from psycopg_pool import ConnectionPool

from main import (
    get_web_docs,
    define_model,
    create_chain,
    create_vector_db,
    create_retriever,
    save_docs,
    define_tool_agent
)
from utils import get_agent_output, get_info_output

BUS_STOPS_URL: str = ("https://pirdop.wordpress.com/%D1%80%D0%B0%D0%B7%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5"
                      "-%D0%BD%D0%B0-%D0%B0%D0%B2%D1%82%D0%BE%D0%B1%D1%83%D1%81%D0%B8/")

if __name__ == "__main__":
    model: ChatGoogleGenerativeAI
    prompt: ChatPromptTemplate
    agent: CompiledStateGraph
    model_pool: ConnectionPool

    docs = get_web_docs(BUS_STOPS_URL)
    model, prompt = define_model()
    chain: Runnable = create_chain(model, prompt)
    vector_db: Chroma = create_vector_db(docs)
    retriever: Runnable = create_retriever(model, prompt, chain, vector_db)
    agent, model_pool = define_tool_agent(retriever, model)
    thread_id: str = str(uuid.uuid4())
    previous_chat: str | None = input("Enter thread id, if you want new chat, press Enter: ")

    save_docs(docs)

    while True:
        required_docs = [BUS_STOPS_URL]
        user_input: str = input("Your message: ")

        if user_input == "":
            continue

        if user_input == "exit":
            model_pool.close()
            break

        # This ensures model memory in a sertan conversation
        config: dict[str, dict] = {"configurable": {"thread_id": previous_chat if previous_chat else thread_id}}

        # This agent provides required information for getting train url
        train_url = get_info_output(user_input)

        if train_url != "":
            required_docs.append(train_url)
            docs = get_web_docs(required_docs)
            vector_db: Chroma = create_vector_db(docs)
            save_docs(docs)

        # This agent gives response to the user
        main_response = get_agent_output(agent, user_input, config)
        print("Answer:", main_response)
