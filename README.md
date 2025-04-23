# semantic-book-recommender with LLM
AI-powered book recommender that suggests books based on your mood, theme, or preferences using semantic search and Gradio UI.

In this project, I have built an AI-powered book recommendation system using Large Language Models (LLMs), vector databases, and a clean Gradio interface. The system allows users to get personalized book recommendations based on natural language queries, sentiment, and classification filters.

---

## 🚀 Project Components

This project is broken down into **five key components**:

1. **🧹 Text Data Cleaning**
   - Notebook: `data-exploration.ipynb`
   - Process and clean raw book data for further analysis and embedding generation.

2. **🔎 Semantic Search & Vector Database**
   - Notebook: `vector-search.ipynb`
   - Learn how to perform semantic (vector) search using LLM-generated embeddings.  
   - Users can query with phrases like _"a book about a person seeking revenge"_ to find relevant books.

3. **🏷️ Zero-Shot Text Classification**
   - Notebook: `text-classification.ipynb`
   - Classify books into categories such as **Fiction** or **Non-Fiction** using zero-shot learning.  
   - Enables filtering based on classification facets.

4. **🎭 Sentiment & Emotion Analysis**
   - Notebook: `sentiment-analysis.ipynb`
   - Extract emotional tones (e.g., joyful, suspenseful, sad) from book descriptions using LLMs.  
   - Users can sort and explore books by mood and tone.

5. **🌐 Web Application with Gradio**
   - Script: `Gradio.py`
   - Deploy a user-friendly web app where users can input their preferences and receive personalized book recommendations powered by semantic search.

---

## 🛠️ Tech Stack & Dependencies

This project was developed using **Python 3.11**.

### Core Libraries:
- `kagglehub`
- `pandas`
- `matplotlib`
- `seaborn`
- `python-dotenv`
- `langchain-community`
- `langchain-opencv`
- `langchain-chroma`
- `transformers`
- `gradio`
- `notebook`
- `ipywidgets`

---
