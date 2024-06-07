import os
import sys
import csv
from dotenv import load_dotenv
import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

sys.path.append(os.path.abspath(os.path.join('..')))

from src.prompt_generation.prompt_generator import generate_prompts
from src.evaluation.evaluator import main as evaluate_prompts

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

def main():
    description = "Explain the concept of RAG in AI"
    scenarios = ["in a business context", "for educational purposes", "for technical documentation"]
    
    # Generate initial prompts
    prompts = generate_prompts(description, scenarios)
    
    # Save prompts to a CSV file
    csv_file = os.path.join(os.path.dirname(__file__), "../data/generated_prompts.csv")
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Prompt"])
        for prompt in prompts:
            writer.writerow([prompt])
    
    # Load URLs from links.csv in the data directory
    links_file = os.path.join(os.path.dirname(__file__), "../data/links.csv")
    with open(links_file, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            url = row[0]
            print(f"\nProcessing URL: {url}")
            
            # Load and index document from the URL
            vectorstore = load_and_index_document(url)
            
            # Retrieve and generate response using RAG
            retriever = vectorstore.as_retriever()
            for prompt in prompts:
                print(f"\nPrompt: {prompt}")
                response = rag_generate(prompt, retriever)
                print("RAG Response:")
    
    # Evaluate the generated prompts
    evaluate_prompts()

if __name__ == "__main__":
    main()
