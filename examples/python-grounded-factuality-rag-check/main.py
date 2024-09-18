import ollama
import warnings
from mattsollamatools import chunker
from newspaper import Article
import numpy as np
import nltk

warnings.filterwarnings(
    "ignore", category=FutureWarning, module="transformers.tokenization_utils_base"
)
nltk.download("punkt", quiet=True)


def getArticleText(url):
    """Gets the text of an article from a URL."""
    article = Article(url)
    article.download()
    article.parse()
    return article.text


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_similarity_search(question_embedding, chunk_embeddings, k=5):
    """Performs cosine similarity search"""
    all_embeddings = [item["embedding"] for article in chunk_embeddings for item in article["embeddings"]]
    source_texts = [item["source"] for article in chunk_embeddings for item in article["embeddings"]]
    question_embedding = np.array(question_embedding).flatten()

    # Compute cosine similarities
    similarities = [cosine_similarity(question_embedding, np.array(emb)) for emb in all_embeddings]

    # Get indices of top k similarities
    top_indices = np.argsort(similarities)[-k:][::-1]

    # Get the source texts of the best matches
    best_matches = [source_texts[idx] for idx in top_indices]

    return best_matches


def process_article(url):
    """Processes the article by downloading, chunking, and embedding it."""
    text = getArticleText(url)
    chunks = chunker(text)

    # Embed (batch) chunks using ollama
    embeddings = ollama.embed(model="all-minilm", input=chunks)["embeddings"]

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        item = {
            "source": chunk,
            "embedding": embedding,
            "sourcelength": len(chunk)
        }
        embedded_chunks.append(item)

    return embedded_chunks


def check(document, claim, model="jmorgan/bespoke-minicheck"):
    """Checks if the claim is supported by the document by calling bespoke-minicheck."""
    prompt = f"Document: {document}\nClaim: {claim}"
    response = ollama.generate(
        model=model, prompt=prompt, options={"num_predict": 2, "temperature": 0.0}
    )
    return response["response"].strip()

if __name__ == "__main__":
    allEmbeddings = []
    default_url = "https://www.theverge.com/2024/9/12/24242439/openai-o1-model-reasoning-strawberry-chatgpt"
    user_input = input(
        "Enter the URL of an article you want to chat with, or press Enter for default example: "
    )
    article_url = user_input.strip() if user_input.strip() else default_url

    # Process the article using the new helper function
    embedded_chunk = process_article(article_url)
    print(f"\nLoaded, chunked, and embedded text from {article_url}.\n")

    while True:
        # Get the question from the user
        question = input("Enter your question or type quit: ")
        if question.lower() == "quit":
            break

        # Embed the user's question using ollama.embed
        question_embedding = ollama.embed(model="all-minilm", input=question)["embeddings"]

        # Perform cosine similarity search to find the best matches
        best_matches = cosine_similarity_search(question_embedding, allEmbeddings, k=4)
        sourcetext = "\n\n".join(best_matches)
        print(f"\nRetrieved chunks: \n{sourcetext}\n")

        # Give the retrieved chunks and question to the chat model
        system_prompt = f"Only use the following information to answer the question. Do not use anything else: {sourcetext}"
        ollama_response = ollama.generate(
            model="llama3.1",
            prompt=question,
            system=system_prompt,
            options={"stream": False},
        )
        answer = ollama_response["response"]
        print(f"LLM Answer:\n{answer}\n")

        # Check each sentence in the response for grounded factuality
        if answer:
            for claim in nltk.sent_tokenize(answer):
                print(f"LLM Claim: {claim}")
                print(
                    f"Is this claim supported by the context according to bespoke-minicheck? {check(sourcetext, claim)}\n"
                )