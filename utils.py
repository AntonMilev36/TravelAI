from langchain_core.messages import AIMessage

from information import get_info
from script import get_train_url


def get_agent_output(agent, user_input, config=None):
    # By documentation the agent has to be runed like this. Ignore this warning
    response: dict[str, list] = agent.invoke(
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

    if not last_ai_message:
        print("AI agent not responding")

    return last_ai_message.content


def get_info_output(user_input):
    info_response = get_info(user_input)

    if info_response[0] == "Not Enough Information":
        return ""

    from_city, to_city, date = info_response
    train_url = get_train_url(from_city, to_city, date)

    return train_url
