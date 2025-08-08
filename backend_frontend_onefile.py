from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
#from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
# add memory in the RAM
from langgraph.checkpoint.memory import MemorySaver
# import streamlit
import streamlit as st

load_dotenv()



class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState) -> ChatState:
    #llm = ChatOpenAI(api_key = OPENAI_API_KEY)
    llm = ChatOpenAI()
    messages = llm.invoke(state["messages"])
    return {"messages": [messages]}

graph = StateGraph(ChatState)
graph.add_node('chat_node', chat_node)
graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=MemorySaver())


# #test the chatbot
# init_state = {'messages': [HumanMessage(content = 'What is the capital of India')]}
# thread_id = 1
#
# config = {'configurable': {'thread_id': thread_id}}
# result = chatbot.invoke(init_state, config = config)
#
# print(result['messages'][-1].content)
# #

# instantiate a empty list if message_history is None
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

#loading the conversation history
# first configure the thread
thread_id = 1
CONFIG = {'configurable': {'thread_id': thread_id}}

for message in st.session_state['message_history']:
    with  st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.text_input('Type here')

if user_input:

    # first add the user message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    #now add the output from chatbot
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]}, config=CONFIG)
    ai_message = response['messages'][-1].content

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
    with st.chat_message('assistant'):
        st.text(ai_message)
