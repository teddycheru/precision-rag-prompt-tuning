import streamlit as st
import os
import csv
import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env file
load_dotenv()

# Ensure required environment variables are set
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")
if not LANGCHAIN_API_KEY:
    raise ValueError("Missing LANGCHAIN_API_KEY in environment variables.")

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Enhanced function to generate prompts
def generate_prompts(description, scenarios, target_audience=None, tone=None, additional_instructions=None):
    prompts = []
    for scenario in scenarios:
        prompt = f"{description} in the context of {scenario}"
        
        if target_audience:
            prompt += f" for {target_audience}"
        
        if tone:
            prompt += f" with a {tone} tone"
        
        if additional_instructions:
            prompt += f". {additional_instructions}"
        
        prompts.append(prompt)
    return prompts

# Function to load, chunk, and index the contents of a web document
def load_and_index_document(url):
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore

# Function to format documents for display
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Function to retrieve and generate using relevant snippets of a blog
def rag_generate(question, retriever):
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)

# Main function to orchestrate prompt generation and retrieval
def main(description, scenarios, target_audience, tone, additional_instructions, url=None, pdf_file=None):
    # Generate initial prompts
    prompts = generate_prompts(description, scenarios, target_audience, tone, additional_instructions)
    
    # Load document from URL or PDF
    if url:
        vectorstore = load_and_index_document(url)
    elif pdf_file:
        # Process the PDF file
        pass
    else:
        st.error("Please provide either a URL or upload a PDF file.")
        return
    
    # Retrieve and generate responses using RAG
    retriever = vectorstore.as_retriever()
    for prompt in prompts:
        st.write(f"\nPrompt: {prompt}")
        try:
            response = rag_generate(prompt, retriever)
        except Exception as e:
            st.error(f"Error generating RAG response: {e}")
        else:
            st.write("RAG Response:")
            st.write(response)

# Streamlit UI
st.title("RAG Response Generator")

description = st.text_input("Enter what you are searching for")
scenarios = st.text_area("Enter scenarios (comma-separated)").split(",")
target_audience = st.selectbox("Select target audience", ["General", "Technical", "Corporate", "Educational"])
tone = st.selectbox("Select tone", ["Formal", "Informal", "Professional", "Friendly"])
additional_instructions = st.text_input("Additional instructions")
url = st.text_input("Enter URL:")

pdf_file = st.file_uploader("or Upload PDF file")

if st.button("Generate Response"):
    main(description, scenarios, target_audience, tone, additional_instructions, url=url, pdf_file=pdf_file)