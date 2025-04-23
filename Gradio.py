import gradio as gr
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

def load_data():
    books = pd.read_csv("books_with_emotions.csv")
    books = books.dropna(subset=['tagged_description'])
    return books

def generate_embeddings(books):
    embedder = OpenAIEmbeddings()
    book_embeddings = embedder.embed_documents(books['tagged_description'].tolist())
    return np.array(book_embeddings), embedder

def recommend_books(user_input, num_recommendations):
    user_embedding = embedder.embed_query(user_input)
    user_embedding = np.array(user_embedding).reshape(1, -1)
    similarities = cosine_similarity(user_embedding, book_embeddings)[0]
    top_indices = similarities.argsort()[-num_recommendations:][::-1]
    results = []
    for idx in top_indices:
        book = books.iloc[idx]
        results.append({
            "Title": book['title'],
            "Author": book['authors'],
            "Category": book['categories'],
            "Description": book['description'][:300] + "...",
            "Thumbnail": book['thumbnail'] if pd.notna(book['thumbnail']) else "https://via.placeholder.com/150"
        })
    return results_display(results)

def results_display(results):
    display = ""
    for book in results:
        display += f"### {book['Title']} by {book['Author']}\n"
        display += f"**Category:** {book['Category']}\n\n"
        display += f"![Cover]({book['Thumbnail']})\n\n"
        display += f"{book['Description']}\n\n---\n"
    return display

def build_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 📖 Semantic Book Recommender")
        gr.Markdown("Describe what you're looking for, and get personalized book recommendations!")

        user_input = gr.Textbox(label="Your Mood / Theme / Keywords", placeholder="e.g., I want a joyful mystery with adventure")
        num_recs = gr.Slider(1, 10, value=5, step=1, label="Number of Recommendations")

        recommend_btn = gr.Button("Recommend Books")
        output = gr.Markdown()

        recommend_btn.click(fn=recommend_books, inputs=[user_input, num_recs], outputs=output)
    return demo

def main():
    global books, book_embeddings, embedder
    load_dotenv()
    books = load_data()
    book_embeddings, embedder = generate_embeddings(books)
    app = build_ui()
    app.launch()

if __name__ == "__main__":
    main()
