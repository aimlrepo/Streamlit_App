from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
#from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
# add memory in the RAM
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    llm = ChatOpenAI()
    messages = llm.invoke(state["messages"])
    return {"messages": [messages]}

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=MemorySaver())


#test the chatbot
init_state = {'messages': [HumanMessage(content = 'What is the capital of India')]}
thread_id = 1

config = {'configurable': {'thread_id': thread_id}}
result = chatbot.invoke(init_state, config = config)

print(result['messages'][-1].content)
#
