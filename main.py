# main.py
"""
Smart Industry Intelligence Pipeline — Final Version
Scrapes all pages (1–11), stores data, summarizes, vectorizes, exposes API + WebSocket.
"""

import os
import json
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Optional

import httpx
import numpy as np
import faiss
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from fastapi import FastAPI, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, String, Integer, DateTime, Text, JSON
from sqlalchemy.orm import sessionmaker, declarative_base

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS as LangFAISS
from langchain_community.docstore.in_memory import InMemoryDocstore

# ------------------------------------------------------------
# ENVIRONMENT & CONFIG
# ------------------------------------------------------------
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not GOOGLE_API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pipeline")

# ------------------------------------------------------------
# FASTAPI INIT
# ------------------------------------------------------------
app = FastAPI(title="Smart Industry Intelligence Pipeline")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------------------------------------------------
# DATABASE (SQLAlchemy)
# ------------------------------------------------------------
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


class Article(Base):
    __tablename__ = "articles"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=False)
    author = Column(String, nullable=True)
    date = Column(String, nullable=True)
    content_body = Column(Text, nullable=True)
    url = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    summary = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)


Base.metadata.create_all(engine)

# ------------------------------------------------------------
# FAISS VECTOR STORE SETUP
# ------------------------------------------------------------
EMBED_DIM = 384  # MiniLM dimensions
os.makedirs("data", exist_ok=True)
FAISS_INDEX_PATH = "data/faiss.index"
FAISS_META_PATH = "data/faiss_meta.json"

embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if os.path.exists(FAISS_INDEX_PATH):
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
else:
    faiss_index = faiss.IndexFlatL2(EMBED_DIM)

if os.path.exists(FAISS_META_PATH):
    with open(FAISS_META_PATH, "r") as f:
        index_to_meta: Dict[str, dict] = json.load(f)
else:
    index_to_meta = {}

vector_store = LangFAISS(
    embedding_function=embedder,
    index=faiss_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)


def save_faiss():
    faiss.write_index(vector_store.index, FAISS_INDEX_PATH)
    with open(FAISS_META_PATH, "w") as f:
        json.dump(index_to_meta, f)


def clean_vec(v: List[float]) -> np.ndarray:
    return np.array([float(x) for x in v], dtype=np.float32)


def add_embedding(text: str, meta: dict):
    try:
        emb = embedder.embed_query(text)
        vec = clean_vec(emb)
        vector_store.index.add(np.array([vec], dtype=np.float32))
        new_id = str(len(index_to_meta))
        index_to_meta[new_id] = meta
        vector_store.index_to_docstore_id[len(index_to_meta)-1] = new_id
        vector_store.docstore.add({new_id: text})
        save_faiss()
    except Exception as e:
        logger.error("FAISS add error: %s", e)


def vector_search(query: str, top_k: int = 5) -> List[dict]:
    if vector_store.index.ntotal == 0:
        logger.warning("FAISS index empty! No results will be returned.")
        return []

    try:
        emb = embedder.embed_query(query)
        vec = clean_vec(emb).reshape(1, -1)
        D, I = vector_store.index.search(vec, top_k)
        results = []
        for idx in I[0]:
            if idx < 0:
                continue
            meta_id = vector_store.index_to_docstore_id.get(idx)
            if not meta_id:
                continue
            meta = index_to_meta.get(meta_id)
            if meta:
                results.append(meta)
        return results
    except Exception as e:
        logger.error("FAISS search error: %s", e)
        return []

# ------------------------------------------------------------
# LLM (Gemini) Setup
# ------------------------------------------------------------
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)


def summarize(text: str) -> str:
    try:
        prompt = f"Summarize this article in 3 concise sentences:\n\n{text}"
        return llm.invoke(prompt).content.strip()
    except Exception as e:
        logger.warning("Summarization failed: %s", e)
        return "Summary unavailable."


def generate_tags(text: str) -> List[str]:
    try:
        prompt = (
            "Extract up to 3 short tags for this article. "
            "Return ONLY a JSON list, e.g. [\"AI\", \"Cloud\", \"Security\"]\n\n" + text
        )
        raw = llm.invoke(prompt).content.strip()
        try:
            tags = json.loads(raw)
        except Exception:
            tags = [t.strip() for t in raw.split(",") if t.strip()]
        return tags[:3]
    except Exception as e:
        logger.warning("Tag generation failed: %s", e)
        return []

# ------------------------------------------------------------
# SCRAPE STATUS GLOBAL
# ------------------------------------------------------------
SCRAPE_STATUS = {
    "running": False,
    "total_found": 0,
    "processed": 0,
    "finished": False
}

# ------------------------------------------------------------
# SCRAPER LOGIC (Pagination 1–11)
# ------------------------------------------------------------
BASE_URL = "https://techreviewer.co/blog"
PAGE_PARAM_BASE = "https://techreviewer.co/blog?979539fa_page={page}"

async def fetch(client: httpx.AsyncClient, url: str) -> str:
    try:
        r = await client.get(url, timeout=20)
        r.raise_for_status()
        return r.text
    except Exception as e:
        logger.error("Fetch failed %s: %s", url, e)
        return ""


async def scrape_article(card, client: httpx.AsyncClient) -> Optional[Dict]:
    try:
        title_tag = card.find("h3", class_="post-card-heading")
        title = title_tag.get_text(strip=True) if title_tag else "N/A"
        relative = card.get("href", "")
        url = "https://techreviewer.co" + relative

        footer = card.find("div", class_="post-card-bottom").find_all("div")
        date = footer[0].get_text(strip=True) if len(footer) else "N/A"
        author = footer[2].get_text(strip=True) if len(footer) > 2 else None

        html = await fetch(client, url)
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("div", class_="blog-post-content")
        content = "\n".join(p.get_text(strip=True) for p in body.find_all("p")) if body else None

        return {
            "title": title,
            "author": author,
            "date": date,
            "url": url,
            "content_body": content
        }
    except Exception as e:
        logger.error("Scrape article failed: %s", e)
        return None


async def scrape_all_cards() -> List[BeautifulSoup]:
    cards = []
    async with httpx.AsyncClient(timeout=20) as client:
        for page in range(1, 12):  # pages 1 to 11 inclusive
            url = PAGE_PARAM_BASE.format(page=page)
            html = await fetch(client, url)
            if not html:
                continue
            soup = BeautifulSoup(html, "html.parser")
            page_cards = soup.find_all("a", class_="post-card w-inline-block")
            if not page_cards:
                continue
            cards.extend(page_cards)
    return cards

# ------------------------------------------------------------
# WEBSOCKET MANAGER
# ------------------------------------------------------------
class WSManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, msg: str):
        to_remove = []
        for ws in self.connections:
            try:
                await ws.send_text(msg)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            self.disconnect(ws)


ws_manager = WSManager()


@app.websocket("/ws/updates")
async def websocket_endpoint(ws: WebSocket):
    await ws_manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(ws)

# ------------------------------------------------------------
# BACKGROUND SCRAPING TASK
# ------------------------------------------------------------
async def process_new_articles():
    SCRAPE_STATUS.update({"running": True, "total_found": 0, "processed": 0, "finished": False})
    session = SessionLocal()
    try:
        cards = await scrape_all_cards()
        SCRAPE_STATUS["total_found"] = len(cards)
        await ws_manager.broadcast(f"Found {len(cards)} articles to process")

        async with httpx.AsyncClient(timeout=20) as client:
            for card in cards:
                a = await scrape_article(card, client)
                if not a or not a.get("content_body"):
                    continue
                # dedupe
                if session.query(Article).filter_by(url=a["url"]).first():
                    continue

                summary = summarize(a["content_body"])
                tags = generate_tags(a["content_body"])

                art = Article(
                    title=a["title"],
                    author=a.get("author"),
                    date=a.get("date"),
                    url=a["url"],
                    content_body=a["content_body"],
                    summary=summary,
                    tags=tags
                )
                session.add(art)
                session.commit()

                add_embedding(a["content_body"], {"title": a["title"], "url": a["url"], "tags": tags})

                SCRAPE_STATUS["processed"] += 1
                await ws_manager.broadcast(f"New article added: {a['title']}")

        await ws_manager.broadcast(f"Scraping finished. Total saved: {SCRAPE_STATUS['processed']}")
    except Exception as e:
        logger.error("Scraping task failed: %s", e)
    finally:
        session.close()
        SCRAPE_STATUS["running"] = False
        SCRAPE_STATUS["finished"] = True

# ------------------------------------------------------------
# API ENDPOINTS
# ------------------------------------------------------------
@app.post("/scrape")
async def start_scrape(background: BackgroundTasks):
    background.add_task(asyncio.run, process_new_articles())
    return {"message": "Scraping started"}

@app.get("/scrape/status")
def get_status():
    return SCRAPE_STATUS

@app.get("/articles")
def get_articles(page: int = 1, limit: int = 10):
    session = SessionLocal()
    total = session.query(Article).count()
    items = session.query(Article).offset((page - 1) * limit).limit(limit).all()
    session.close()

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "articles": [
            {
                "title": a.title,
                "author": a.author,
                "date": a.date,
                "summary": a.summary,
                "tags": a.tags,
                "url": a.url,
            }
            for a in items
        ]
    }

class SearchQuery(BaseModel):
    query: str

@app.post("/search")
def search_articles(q: SearchQuery, top_k: int = 5):
    res = vector_search(q.query, top_k)
    return {"query": q.query, "results": res}
