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

class VectorIndexBackend:
    """
    Pluggable backend interface for vector indexing.
    Implementations must provide setup/add/search/count/reset.
    """
    def setup(self, dim: int):
        raise NotImplementedError

    def add(self, vec, meta) -> bool:
        """Add a single vector with metadata. Return True if kept (not deduped)."""
        raise NotImplementedError

    def search(self, query_vec, k: int):
        """Return list of dicts with metadata and a 'score' field."""
        raise NotImplementedError

    def count(self) -> int:
        return 0

    def reset(self):
        pass


class FaissInMemoryBackend(VectorIndexBackend):
    """
    Default in-memory FAISS backend with HNSW and on-the-fly dedup.
    """
    def __init__(self, dup_threshold: float = 0.94, m: int = 32, ef_search: int = 64, ef_construction: int = 200):
        self._deduper = None
        self._metas = []
        self.index = None  # exposed for compatibility
        self._params = dict(dup_threshold=dup_threshold, m=m, ef_search=ef_search, ef_construction=ef_construction)
        self._rep_to_dups = {}

    def setup(self, dim: int):
        if self._deduper is not None:
            return
        # Initialize deduper (which initializes FAISS HNSW index internally)
        self._deduper = FaissDedupWrapper(dim=dim,
                                          m=self._params["m"],
                                          ef_search=self._params["ef_search"],
                                          ef_construction=self._params["ef_construction"],
                                          dup_threshold=self._params["dup_threshold"])
        self.index = self._deduper.index

    def add(self, vec, meta) -> bool:
        kept, rep_row_id = self._deduper.add_with_dedup(vec, row_id=len(self._metas))
        if kept:
            self._metas.append(meta)
        else:
            # Track duplicate mapping
            self._rep_to_dups = self._deduper.rep_to_dups
        return kept

    def search(self, query_vec, k: int):
        scores, idxs = self.index.search(query_vec[None, :], k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._metas):
                continue
            meta = dict(self._metas[idx])
            meta["score"] = float(score)
            results.append(meta)
        return results

    def count(self) -> int:
        return len(self._metas)

    def reset(self):
        self._deduper = None
        self._metas = []
        self.index = None
        self._rep_to_dups = {}

    def get_duplicates_map(self):
        return dict(self._rep_to_dups)


class LocalEmbeddingContext():
    def __init__(self, mc_query=None, mc_window=30, index_backend: VectorIndexBackend = None):
        self.config = embedding_config()
        print(self.config)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.embedding_model = SentenceTransformer("google/embeddinggemma-300m", 
                                token = self.config.hf_token,
                                device="mps")
        
        # Pluggable backend
        self.backend: VectorIndexBackend = index_backend or FaissInMemoryBackend()
        self.index = None  # legacy attribute mirrors backend.index if present
        self.cdocs = pd.DataFrame()
        # Attention-over-time state
        self._story_date_counts = {}
        self._id_to_date = {}
        self._kept_story_ids = set()
        
        if mc_query != None:
            self.build_index_from_query(mc_query, mc_window)
    
    def build_index_from_query(self, query, window, batch_size=16, progress_callback=None):
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
                normalize_embeddings=True,
                show_progress_bar=False
            ).astype("float32")
            # Initialize backend and index on first batch
            if self.index is None:
                self.backend.setup(dim=emb.shape[1])
                self.index = getattr(self.backend, "index", None)
            for v, meta in zip(emb, buf_meta):
                if self.backend.add(v, meta):
                    num_kept += 1
            buf_inputs.clear()
            buf_meta.clear()
            if progress_callback is not None:
                try:
                    progress_callback({
                        "stories": num_stories,
                        "chunks": num_chunks,
                        "kept": num_kept
                    })
                except Exception:
                    pass

        # Reset attention state
        from collections import defaultdict
        self._story_date_counts = defaultdict(int)
        self._id_to_date = {}
        self._kept_story_ids = set()

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
                story_id = story.get("stories_id") or story.get("id") or None
                # Track original attention counts by date (date-only)
                if publish_date:
                    try:
                        dstr = str(publish_date)[:10]
                        self._story_date_counts[dstr] += 1
                        if story_id is not None:
                            self._id_to_date[story_id] = dstr
                    except Exception:
                        pass
                chunks = self.chunk_text(text)
                for ci, ch in enumerate(chunks):
                    num_chunks += 1
                    buf_inputs.append(self.doc_prompt(title, ch["text"]))
                    buf_meta.append({
                        "title": title,
                        "text": ch["text"],
                        "publish_date": publish_date,
                        "story_id": story_id,
                        "chunk_index": ci,
                        "para_start": ch.get("para_start"),
                        "para_end": ch.get("para_end"),
                    })
                    # Flush when buffer large enough
                    if len(buf_inputs) >= 128:
                        flush_buffer()

            # Flush between pages to bound memory
            flush_buffer()
            if progress_callback is not None:
                try:
                    progress_callback({
                        "stories": num_stories,
                        "chunks": num_chunks,
                        "kept": num_kept
                    })
                except Exception:
                    pass

        # Final flush
        flush_buffer()

        # Materialize DataFrame of kept rows for downstream utilities (if backend supports it)
        try:
            kept_count = self.backend.count()
        except Exception:
            kept_count = 0
        if kept_count > 0 and hasattr(self.backend, "_metas"):
            self.cdocs = pd.DataFrame(self.backend._metas)
            # Track semantic attention by story id if present
            if "story_id" in self.cdocs.columns:
                for sid in self.cdocs["story_id" ].dropna().unique().tolist():
                    try:
                        self._kept_story_ids.add(sid)
                    except Exception:
                        pass
        else:
            self.cdocs = pd.DataFrame(columns=["title", "text", "publish_date"])

        print(f"Built index: stories={num_stories}, chunks={num_chunks}, kept={num_kept} in {datetime.datetime.now()-start}")
        if progress_callback is not None:
            try:
                progress_callback({
                    "stories": num_stories,
                    "chunks": num_chunks,
                    "kept": num_kept,
                    "done": True
                })
            except Exception:
                pass

    def attention_over_time(self):
        """
        Returns a DataFrame with columns: date, original, semantic.
        original: count of all stories per date for the original query.
        semantic: count of unique kept stories per date (derived from indexed chunks).
        """
        # Build semantic counts by mapping kept story ids to dates
        from collections import defaultdict
        sem_counts = defaultdict(int)
        for sid in self._kept_story_ids:
            dstr = self._id_to_date.get(sid)
            if dstr:
                sem_counts[dstr] += 1
        # Union of dates
        all_dates = set(self._story_date_counts.keys()) | set(sem_counts.keys())
        if not all_dates:
            return pd.DataFrame(columns=["date", "original", "semantic"]).astype({"date": "datetime64[ns]"})
        rows = []
        for d in sorted(all_dates):
            rows.append({
                "date": d,
                "original": int(self._story_date_counts.get(d, 0)),
                "semantic": int(sem_counts.get(d, 0)),
            })
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.sort_values("date")
        return df


    def chunk_text(self, txt, max_tokens = 900):
        """
        "Chunk" text blocks - encode them and split them into blocks that can be consumed by the embedding model

        """
        paras = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
        # Track paragraph offsets for citations
        chunks, current, cur_tokens = [], [], 0
        current_start_para = 0
        for i, p in enumerate(paras):
            t = len(self.encoder.encode(p))
            if cur_tokens + t > max_tokens and current:
                chunk_text = "\n\n".join(current)
                chunks.append({
                    "text": chunk_text,
                    "para_start": current_start_para,
                    "para_end": i - 1
                })
                current, cur_tokens = [p], t
                current_start_para = i
            else:
                current.append(p); cur_tokens += t
    
        if current:
            chunk_text = "\n\n".join(current)
            chunks.append({
                "text": chunk_text,
                "para_start": current_start_para,
                "para_end": len(paras) - 1
            })

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
        results = self.backend.search(qv.astype("float32"), k)
        # Attach stable chunk_id for citations
        for r in results:
            sid = r.get("story_id")
            cix = r.get("chunk_index")
            if sid is not None and cix is not None:
                r["chunk_id"] = f"{sid}:{cix}"
        rows = pd.DataFrame(results)
        if rows.empty:
            return rows
        return rows.sort_values("score", ascending=False)

    def qa(self, query, k=20):
        q= f"task: question answering | query: {query}"
        qv = self.embedding_model.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        results = self.backend.search(qv.astype("float32"), k)
        for r in results:
            sid = r.get("story_id")
            cix = r.get("chunk_index")
            if sid is not None and cix is not None:
                r["chunk_id"] = f"{sid}:{cix}"
        rows = pd.DataFrame(results)
        if rows.empty:
            return rows
        return rows.sort_values("score", ascending=False)

    def count(self) -> int:
        try:
            return self.backend.count()
        except Exception:
            return 0

    # --- Citation helpers ---
    def get_chunk_metadata(self, chunk_ids):
        """
        Resolve chunk_ids like "story_id:chunk_index" to metadata rows.
        """
        if not hasattr(self.backend, "_metas"):
            return []
        want = set(chunk_ids)
        out = []
        for m in getattr(self.backend, "_metas", []):
            sid = m.get("story_id")
            cix = m.get("chunk_index")
            if sid is None or cix is None:
                continue
            cid = f"{sid}:{cix}"
            if cid in want:
                out.append(dict(m, chunk_id=cid))
        return out

    def get_snippet(self, chunk_id, max_chars=240):
        rows = self.get_chunk_metadata([chunk_id])
        if not rows:
            return ""
        text = (rows[0].get("text") or "").strip()
        return text[:max_chars]

    def build_citation_bundle(self, results_df):
        """
        Build a citation mapping and reference list from a search results DataFrame.
        Returns { citations: [{id:"[1]", story_id, title, url?, publish_date, snippet}], inline_map: {chunk_id:"[1]"} }
        """
        if results_df is None or results_df.empty:
            return {"citations": [], "inline_map": {}}
        # Group by story_id to assign numeric ids
        refs = []
        inline_map = {}
        story_to_num = {}
        next_id = 1
        for _, row in results_df.iterrows():
            sid = row.get("story_id")
            cix = row.get("chunk_index")
            cid = row.get("chunk_id") or (f"{sid}:{cix}" if sid is not None and cix is not None else None)
            if sid is None or cid is None:
                continue
            if sid not in story_to_num:
                label = f"[{next_id}]"
                story_to_num[sid] = label
                refs.append({
                    "id": label,
                    "story_id": sid,
                    "title": row.get("title"),
                    "publish_date": row.get("publish_date"),
                    "snippet": (row.get("text") or "").strip()[:240],
                })
                next_id += 1
            inline_map[cid] = story_to_num[sid]
        return {"citations": refs, "inline_map": inline_map}

    def get_duplicates_map(self):
        if hasattr(self.backend, "get_duplicates_map"):
            return self.backend.get_duplicates_map()
        return {}