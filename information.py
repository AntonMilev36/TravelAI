import datetime
import os
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL: str = "gemini-2.5-flash"


def current_date() -> Any:
    date = datetime.datetime.now()

    return date.strftime("%d.%m.%Y")


def get_info(user_input: str):
    llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )

    prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Get information about user travel plan. You need to provide, separated by comma, city "
                    "the user want to go from, the city user wants to go to and the date, in this "
                    "order. If date is not provided get today day. The date format need to be "
                    "'date.month.year' (example: 18.04.2024). The names of both cities in the "
                    "list need to be writen in Bulgarian with all letters in uppercase. "
                    "Use this to know the current date: {date}"
                    "If the text don't provide any cities or provide just one or more that 2 cities "
                    "or not travel question at all, just provide 'Not Enough Information' as answer. "
                    "Be very patient with the cities names, no additional or missing characters"
                ),
                ("human", "{input}")
            ]
        )

    parser = CommaSeparatedListOutputParser()

    chain: Runnable = prompt | llm | parser

    resource: Any = chain.invoke(
        {
            "input": user_input,
            "date": current_date()
        }
    )

    return resource
