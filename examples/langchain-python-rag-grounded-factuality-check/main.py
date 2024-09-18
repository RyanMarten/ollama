import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader

# Create a sample document
sample_text = """
Artificial Intelligence (AI) is a broad field of computer science focused on creating intelligent machines that can perform tasks that typically require human intelligence. These tasks include visual perception, speech recognition, decision-making, and language translation. AI systems are designed to analyze their environment and take actions that maximize their chance of achieving specific goals.

Machine Learning (ML) is a subset of AI that focuses on the development of algorithms and statistical models that enable computer systems to improve their performance on a specific task through experience. Instead of explicitly programming rules, ML algorithms learn patterns from data.

Deep Learning is a subfield of machine learning that uses artificial neural networks with multiple layers (hence "deep") to progressively extract higher-level features from raw input. For example, in image processing, lower layers might identify edges, while higher layers might identify concepts relevant to a human such as digits or letters or faces.
"""

# Save the sample text to a file
with open("sample_document.txt", "w") as f:
    f.write(sample_text)

# Load the document
loader = TextLoader("sample_document.txt")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Create embeddings and store them in a vector database
embeddings = OllamaEmbeddings(model="llama3.1")
vectorstore = FAISS.from_documents(texts, embeddings)

# Create a retrieval-based question-answering chain
llm = Ollama(model="llama3.1")
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Example usage
query = "What is the main topic of the document?"
result = qa.run(query)
print(result)

# Clean up the sample file
os.remove("sample_document.txt")
