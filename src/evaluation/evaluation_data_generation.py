import os
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

# Function to create evaluation data
def create_evaluation_data(description, scenarios, blog_url):
    vectorstore = load_and_index_document(blog_url)
    retriever = vectorstore.as_retriever()
    
    evaluation_data = []
    
    prompts = generate_prompts(description, scenarios)
    for prompt in prompts:
        response = rag_generate(prompt, retriever)
        evaluation_data.append({
            "description": description,
            "scenario": prompt,
            "blog_url": blog_url,
            "generated_response": response
        })
    
    return evaluation_data

# Save evaluation data to CSV
def save_evaluation_data_to_csv(evaluation_data, file_path):
    keys = evaluation_data[0].keys()
    with open(file_path, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(evaluation_data)

# Main function to orchestrate prompt generation and retrieval
def main():
    description = "Explain the concept of RAG in AI"
    scenarios = ["in a business context", "for educational purposes", "for technical documentation"]
    target_audience = "non-experts"
    tone = "friendly"
    additional_instructions = "Include examples where relevant."
    
    # Generate initial prompts
    prompts = generate_prompts(description, scenarios, target_audience, tone, additional_instructions)
    
    # Load URLs from links.csv in the data directory
    csv_file = os.path.join(os.path.dirname(__file__), "../../data/links.csv")
    with open(csv_file, "r", newline="") as file:
        reader = csv.reader(file)
        for row in reader:
            url = row[0]
            print(f"\nProcessing URL: {url}")
            
            # Load and index document from the URL
            try:
                vectorstore = load_and_index_document(url)
            except Exception as e:
                print(f"Error loading and indexing document from URL {url}: {e}")
                continue
            
            # Retrieve and generate response using RAG
            retriever = vectorstore.as_retriever()
            evaluation_data = []  # Initialize evaluation data list
            for prompt in prompts:
                print(f"\nPrompt: {prompt}")
                try:
                    response = rag_generate(prompt, retriever)
                    print("RAG Response:")
                    print(response)
                    evaluation_data.append({
                        "description": description,
                        "scenario": prompt,
                        "blog_url": url,
                        "generated_response": response
                    })  # Add data to evaluation_data list
                except Exception as e:
                    print(f"Error generating RAG response for prompt '{prompt}': {e}")
            
            # Save evaluation data to CSV
            eval_data_csv_path = os.path.join(os.path.dirname(__file__), "../../data/evaluation_data.csv")
            save_evaluation_data_to_csv(evaluation_data, eval_data_csv_path)  # Save data to CSV

if __name__ == "__main__":
    main()


