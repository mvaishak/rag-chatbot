import pymupdf
import re
import spacy
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import chainlit as cl
import os
from langchain.prompts import PromptTemplate
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import phoenix as px
from phoenix.trace import using_project
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register

session = px.launch_app()
tracer_provider = register(project_name="chainlit_rag", endpoint="http://localhost:6006/v1/traces")

LangChainInstrumentor().instrument(tracer_provider=tracer_provider)


# Load spaCy model for text processing
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Text cleaning function
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text.lower().strip()

# Intelligent text splitting
def split_into_chunks(text, max_length=1000):
    doc = nlp(text)
    chunks = []
    current_chunk = []
    current_length = 0
    for sent in doc.sents:
        if current_length + len(sent) > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sent.text)
        current_length += len(sent)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Load and preprocess the PDF
pdf_path = r"C:\Users\vaish\Downloads\Documents\ExplorersGuide.pdf"
pdf_text = extract_text_from_pdf(pdf_path)
cleaned_text = clean_text(pdf_text)
text_chunks = split_into_chunks(cleaned_text)

# Initialize improved models
embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Create embeddings
embeddings = embedding_model.encode(text_chunks)

d = embeddings.shape[1]
index = faiss.IndexFlatL2(d)
index.add(np.array(embeddings).astype('float32'))

def retrieve_similar_chunks(query, k=3):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding.astype('float32'), k)
    return [text_chunks[i] for i in indices[0]]

# Initialize Azure OpenAI model for evaluation
azure_llm = AzureChatOpenAI(
    azure_deployment="llmops_CT_GPT4o",
    openai_api_version="2023-07-01-preview",
    azure_endpoint="Enter Endpoint",
    api_key="Enter API Key",
    temperature=0
)

# Create a chain for the QA process
template = """
Given the following context and question, provide a relevant answer:

Context: {context}

Question: {question}

Answer:
"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
qa_chain = prompt | azure_llm | StrOutputParser()


@cl.on_message
async def main(message: cl.Message):
    query = message.content
    
    relevant_chunks = retrieve_similar_chunks(query)
    context = " ".join(relevant_chunks)
    response = qa_chain.invoke({"context": context, "question": query})
    
    await cl.Message(content=response).send()


if __name__ == "__main__":
    cl.run()