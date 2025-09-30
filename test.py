import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

user_input = input("Enter key words: ")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.9
)

image_llm = InferenceClient(
    "stabilityai/stable-diffusion-xl-base-1.0",
    token=os.getenv("HUD_FACE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Your task is to take the words provided from the user and"
                   "make them in text that AI models can use for making funny pictures, "
                   "not only parsing them next each other, only one option"),
        ("human", "{input}")
    ]
)

chain = prompt | llm

description = chain.invoke(
    {
        "input": user_input
    }
)

image = image_llm.text_to_image(description.content)

image.save(os.path.expanduser("~/Desktop/output.png"))

print("Task completed :)")
