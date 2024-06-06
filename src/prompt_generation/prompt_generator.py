import os
from dotenv import load_dotenv
import bs4
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

# Function to generate prompts
def generate_prompts(description, scenarios):
    prompts = []
    for scenario in scenarios:
        prompt = f"{description} in the context of {scenario}"
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
def main():
    description = "Explain the concept of RAG in AI"
    scenarios = ["in a business context", "for educational purposes", "for technical documentation"]
    
    # Generate initial prompts
    prompts = generate_prompts(description, scenarios)
    
    # Example: Load and index a document from the web
    url = "https://lilianweng.github.io/posts/2023-06-23-agent/"
    vectorstore = load_and_index_document(url)
    
    # Retrieve and generate a response using RAG
    retriever = vectorstore.as_retriever()
    response = rag_generate("What is Task Decomposition?", retriever)
    
    print("Generated Prompts:")
    for prompt in prompts:
        print(prompt)
    
    print("\nRAG Response:")
    print(response)

if __name__ == "__main__":
    main()
