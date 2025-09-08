from mediacloud import api
import datetime 
import pandas as pd
from sentence_transformers import SentenceTransformer
import tiktoken
import re


from FaissDedup import FaissDedupWrapper

from pydantic_settings import BaseSettings, SettingsConfigDict

US_NATIONAL_COLLECTION = 34412234 #just a default for now

class embedding_config(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    mc_api_key: str = ""
    hf_token: str = ""

class LocalEmbeddingContext():
    def __init__(self, mc_query=None, mc_window=30):
        self.config = embedding_config()
        print(self.config)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.embedding_model = SentenceTransformer("google/embeddinggemma-300m", 
                                token = self.config.hf_token,
                                device="mps")
        
        # Vector index + metadata built incrementally
        self.index = None  # kept for compatibility; will reference self.deduper.index
        self.deduper = None  # FaissDedupWrapper, initialized on first batch when dim known
        self._kept_rows = []  # list of metadata dicts aligned with FAISS internal ids
        self.cdocs = pd.DataFrame()
        
        if mc_query != None:
            self.build_index_from_query(mc_query, mc_window)
    
    def build_index_from_query(self, query, window, batch_size=16):
        """
        Stream stories page-by-page from MediaCloud, chunk, embed in small batches,
        and add with on-the-fly dedup to an incremental FAISS index.
        """
        start = datetime.datetime.now()
        today = datetime.date.today()
        window = datetime.timedelta(window)
        mc_search = api.SearchApi(self.config.mc_api_key)

        pagination_token = None
        more_stories = True

        num_stories = 0
        num_chunks = 0
        num_kept = 0

        # Small staging buffers per page to batch embeds
        buf_inputs = []
        buf_meta = []

        def flush_buffer():
            nonlocal num_kept
            if not buf_inputs:
                return
            emb = self.embedding_model.encode(
                buf_inputs,
                batch_size = min(batch_size, 16),
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype("float32")
            # Initialize deduper and index on first batch
            if self.deduper is None:
                self.deduper = FaissDedupWrapper(dim=emb.shape[1])
                self.index = self.deduper.index  # maintain legacy attribute
            for v, meta in zip(emb, buf_meta):
                kept = self.deduper.add_with_dedup(v, row_id=len(self._kept_rows))
                if kept:
                    self._kept_rows.append(meta)
                    num_kept += 1
            buf_inputs.clear()
            buf_meta.clear()

        while more_stories:
            page, pagination_token = mc_search.story_list(
                query,
                collection_ids=[US_NATIONAL_COLLECTION],
                start_date = today-window,
                end_date=today,
                pagination_token = pagination_token,
                expanded = True
            )
            more_stories = pagination_token is not None

            for story in page:
                num_stories += 1
                text = story.get("text") or ""
                title = story.get("title") or ""
                publish_date = story.get("publish_date")
                chunks = self.chunk_text(text)
                for ch in chunks:
                    num_chunks += 1
                    buf_inputs.append(self.doc_prompt(title, ch))
                    buf_meta.append({
                        "title": title,
                        "text": ch,
                        "publish_date": publish_date
                    })
                    # Flush when buffer large enough
                    if len(buf_inputs) >= 128:
                        flush_buffer()

            # Flush between pages to bound memory
            flush_buffer()

        # Final flush
        flush_buffer()

        # Materialize DataFrame of kept rows for downstream utilities
        if self._kept_rows:
            self.cdocs = pd.DataFrame(self._kept_rows)
        else:
            self.cdocs = pd.DataFrame(columns=["title", "text", "publish_date"])

        print(f"Built index: stories={num_stories}, chunks={num_chunks}, kept={num_kept} in {datetime.datetime.now()-start}")


    def chunk_text(self, txt, max_tokens = 900):
        """
        "Chunk" text blocks - encode them and split them into blocks that can be consumed by the embedding model

        """
        paras = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
        chunks, current, cur_tokens = [], [], 0
        for p in paras:
            t = len(self.encoder.encode(p))
            if cur_tokens + t > max_tokens and current:
                chunks.append("\n\n".join(current))
                current, cur_tokens = [p], t
            else:
                current.append(p); cur_tokens += t
    
        if current: chunks.append("\n\n".join(current))

        return chunks

    def doc_prompt(self, title, text):
        t = (title or "none").strip()[:300] #keep title short
        return f'title: {t} | text: {text}'
    

    def embed_and_index(self):
        """
        Legacy API retained for compatibility; no-op because indexing is now streaming.
        """
        return


    
    def search(self, query, k = 20):
        if self.index == None:
            raise RuntimeError
            
        q= f"task: search result | query: {query}"
        qv = self.embedding_model.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(qv[None, :], k)
        # Map FAISS internal ids -> kept metadata rows (aligned by insertion order)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._kept_rows):
                continue
            meta = dict(self._kept_rows[idx])
            meta["score"] = float(score)
            results.append(meta)
        rows = pd.DataFrame(results)
        if rows.empty:
            return rows
        return rows.sort_values("score", ascending=False)

    def qa(self, query, k=20):
        q= f"task: question answering | query: {query}"
        qv = self.embedding_model.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(qv[None, :], k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._kept_rows):
                continue
            meta = dict(self._kept_rows[idx])
            meta["score"] = float(score)
            results.append(meta)
        rows = pd.DataFrame(results)
        if rows.empty:
            return rows
        return rows.sort_values("score", ascending=False)