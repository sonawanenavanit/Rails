import streamlit as st
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)


search= DuckDuckGoSearchRun(name="Search")
api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
wiki.name


st.title("Langchain - Chat with search")

st.sidebar.title("Setting")
api_key = st.sidebar.text_input("API Key",type="password")

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assisstant","content":"Hi, I'm a chatbot who can serch the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt:=st.chat_input(placeholder="What is machine learning") and api_key:
    st.session_state.messages.append({"role":"user","content":prompt})

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,handling_parsing_error=True)
    
    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({"role":"assistant","content":response})
        st.write(response)