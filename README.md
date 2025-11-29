# Smart Industry Intelligence Pipeline

This project is a **fully automated backend system** that scrapes technical blog posts, summarizes them using an LLM (Google Gemini), categorizes them with tags, and exposes the data via a FastAPI REST API and WebSocket notifications.

---

## Features

- Scrapes articles from **TechReviewer** (pages 1–11)
- Generates **summaries** and **tags** using LLM
- Stores metadata in **PostgreSQL**
- Stores **vector embeddings** in FAISS for semantic search
- Provides endpoints:
  - `GET /articles` – list all articles
  - `POST /search` – semantic search using vector embeddings
  - `GET /scrape/status` – check scraping progress
  - `POST /scrape` – start scraping articles
- **WebSocket** `ws://localhost:8000/ws/updates` notifications for real-time updates

---

## Prerequisites

- Docker & Docker Compose installed
- `.env` file with the following variables:

```env
GOOGLE_API_KEY=your_google_gemini_api_key
DATABASE_URL=postgresql+psycopg2://<POSTGRES_USER>:<POSTGRES_PASSWORD>@db:5432/<POSTGRES_DB>
POSTGRES_USER=your_username
POSTGRES_PASSWORD=your_password
POSTGRES_DB=your_db_name
