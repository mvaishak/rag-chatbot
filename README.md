# RAG Chatbot with Chainlit and Phoenix Tracing

This project implements a Retrieval-Augmented Generation (RAG) chatbot using Chainlit, integrating open-source models, vector database, and Phoenix tracing for performance monitoring.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Components](#components)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Phoenix Tracing](#phoenix-tracing)
7. [Video Demo](#video-demo)

## Project Overview

This RAG chatbot is designed to answer questions based on the content of a PDF document. It uses advanced NLP techniques for text processing, embedding generation, and question answering. In this example, I am providing the game guide for Zelda BOTW. This chatbot will help answer any questions regarding the game using the guide PDF.

## Installation

1. Create a virtual environment:
   ```
   conda create -n rag_chatbot_env python=3.9
   conda activate rag_chatbot_env
   ```

2. Install dependencies:
   ```
   pip install pymupdf spacy transformers sentence_transformers faiss-cpu chainlit langchain langchain_openai phoenix arize-phoenix openinference
   python -m spacy download en_core_web_sm
   ```

3. Set up Azure OpenAI credentials (ensure you have access to Azure OpenAI services).

## Usage

1. Place your PDF file in the specified directory and update the `pdf_path` variable in the script.
2. Run the Chainlit app:
   ```
   chainlit run app.py
   ```
3. Open the provided URL in your browser to interact with the chatbot.

## Components

1. **PDF Text Extraction**: Uses PyMuPDF to extract text from the PDF.
2. **Text Processing**: Utilizes spaCy for text cleaning and intelligent chunking.
3. **Embedding Generation**: Uses SentenceTransformer model 'all-mpnet-base-v2' for creating embeddings.
4. **Vector Database**: Implements FAISS for efficient storage and retrieval of embeddings.
5. **Question Answering**: Utilizes Azure OpenAI's GPT model for generating responses.
6. **Chainlit Interface**: Provides a user-friendly chat interface.

## Phoenix Tracing

Phoenix tracing is integrated to monitor the chatbot's performance:

1. Launches a Phoenix app session.
2. Registers a tracer provider for the "chainlit_rag" project.
3. Instruments LangChain for detailed tracing.

Access the Phoenix UI at `http://localhost:6006` to view real-time traces and performance metrics.

## Video Demo

[https://youtu.be/DC6WQwzZbG0]

## Future Improvements

1. Implement explicit confidence scoring for answers.
2. Enhance error handling and user feedback.
3. Optimize chunk retrieval and context building.

