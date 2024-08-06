from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_chroma import Chroma
load_dotenv()


os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


api_key =st.text_input("Enter you api key:" ,type="password")

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")

if api_key:
    llm=ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

    session_id=st.text_input("SessionId", value="default_session")

    if "store" not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose A PDF file",type="pdf",accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader = PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

    # Splitter and created embedding for the document
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        splits=text_splitter.split_documents(documents)
        vectorestore = Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever = vectorestore.as_retriever()
        retriever
    
        contextualize_q_system_prompt=(
            """
            Given a chat history and latest user question
            which might refence context in the chat history
            formulate a standalone question which can be understood
            without the chat history. do not answer the question
            just reformulate it if needed and otherwise return as it is.
            """
        )
        contextualize_q_system_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriver = create_history_aware_retriever(llm,retriever,contextualize_q_system_prompt)

        # Prompt Template

        system_prompt = (
            "You are an assistant for question-answering task"
            "Use the following pieces of retrived context to answer"
            "the question."
            "\n\n"
            "{context}"
        )

        qa_promt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        question_answer_chain=create_stuff_documents_chain(llm,qa_promt)
        rag_chain=create_retrieval_chain(history_aware_retriver,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_message_key="input",
            history_message_key="chat_history",
            output_message_key="answer"
        )

        user_input = st.text_input("Your question:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config = {
                    "configurable":{ "session_id":session_id}
                }
            )
            st.write(st.session_state.store)
            st.success("Assistant", response['answer'])
            st.write("Chat History", session_history.messages)

else:
    st.warning("Please Enter the groq API key")










