# Precision_RAG_Prompt_Tuning

## Overview

Precision RAG Prompt Tuning is a project aimed at enhancing the performance of the Retrieval-Augmented Generation (RAG) model through advanced prompt tuning techniques. The project involves automatically generating prompts based on user queries and specific scenarios, retrieving relevant information from web documents, and incorporating external sources such as blog links and PDF data to provide comprehensive responses. By evaluating the quality and accuracy of the generated responses, this project seeks to refine the RAG model's ability to deliver precise and contextually relevant information, thereby improving its overall effectiveness.

## Features

- **Automatic Prompt Generation**: The system automatically generates prompts based on a given description, scenarios, target audience, tone, and additional instructions.

- **Web Document Retrieval**: Web documents are fetched from URLs provided in a CSV file, and relevant information is extracted using language embeddings and document indexing.

- **RAG Model Integration**: The Retrieval-Augmented Generation (RAG) model is utilized to generate responses based on the prompts and retrieved information.

- **Evaluation Data Generation**: Evaluation data is automatically generated by comparing the generated responses with ground truth responses, allowing for performance evaluation.

## Requirements

- Python 3.7+
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
https://github.com/teddycheru/Precision_RAG_Prompt_Tuning.git

2. Navigate to the project directory:
   `cd precision-rag-prompt-tuning`

3. Install dependencies: `pip install -r requirements.txt`

4. Set up environment variables by creating a `.env` file and adding the required keys:
   `OPENAI_API_KEY=your_openai_api_key`
`LANGCHAIN_API_KEY=your_langchain_api_key`
 
## Usage

1. Modify the parameters in `evaluation_data_generation.py` as needed, such as description, scenarios, target audience, tone, additional instructions, and ground truth file.

2. Run the script: `python3 src/main.py`

3. Evaluation data will be generated and saved in `/data/evaluation_data.csv`.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure that your code follows the project's coding style and conventions.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenAI](https://openai.com) for providing the GPT-3 model and API.
- [LangChain](https://langchain.io) for providing language processing tools and APIs.



   






