from mediacloud import api
import datetime 
import time
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import tiktoken
import os
import re
import faiss
from collections import defaultdict

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
        
        self.raw_stories = pd.DataFrame()                 
        if mc_query != None:
            self.get_stories(mc_query, mc_window)
            
        
        self.index = None
        self.cdocs = pd.DataFrame()
        if not self.raw_stories.empty:
            self.embed_and_index()
            #self.dedup_index()
    
    def get_stories(self, query, window):
        start = datetime.datetime.now() 
        today = datetime.date.today()
        window = datetime.timedelta(window)
        mc_search = api.SearchApi(self.config.mc_api_key)

    
        all_stories = []
        pagination_token = None
        more_stories = True
        
        while more_stories:
            page, pagination_token = mc_search.story_list(query, collection_ids=[US_NATIONAL_COLLECTION], 
                                                          start_date = today-window, end_date=today,
                                                          pagination_token = pagination_token, expanded = True)
            all_stories += page
            more_stories = pagination_token is not None
    
        print(f"Got {len(all_stories)} stories in {datetime.datetime.now()-start}")
        self.raw_stories = pd.DataFrame.from_records(all_stories)


    def chunk_text(self, txt, max_tokens = 900):
        """
        "Chunk" text blocks - encode them and split them into blocks that can be consumed by the embedding model

        """
        paras = [p.strip() for p in re.split(r"\n{2,}", txt) if p.strip()]
        chunks, current, cur_tokens = [], [], 0
        for p in paras:
            t = len(self.encoder.encode(p))
            if cur_tokens + t > max_tokens and current:
                chunks.append("/n/n".join(current))
                current, cur_tokens = [p], t
            else:
                current.append(p); cur_tokens += t
    
        if current: chunks.append("/n/n".join(current))

        return chunks

    def doc_prompt(self, title, text):
        t = (title or "none").strip()[:300] #keep title short
        return f'title: {t} | text: {text}'
    

    def embed_and_index(self):
        """
        Run the fetched text through the embedding model and store it in a vector index. 
        """
        start = datetime.datetime.now() 
        if self.raw_stories.empty:
            raise RuntimeError

        records = []
        for _, row in self.raw_stories.iterrows():
            chunks = self.chunk_text(row["text"])
            for i, ch in enumerate(chunks):
                records.append({
                    "title":row["title"], "text":ch, "publish_date":row["publish_date"]})

        self.cdocs = pd.DataFrame(records)
        
        doc_inputs = [self.doc_prompt(t, x) for t, x in zip(self.cdocs["title"], self.cdocs["text"])]
        doc_emb = self.embedding_model.encode(doc_inputs, 
                                              batch_size = 16, 
                                              convert_to_numpy=True, 
                                              normalize_embeddings=True).astype("float32")
        
        d = doc_emb.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(doc_emb)
        print(f"Created index in {datetime.datetime.now()-start}")


    
    def search(self, query, k = 20):
        if self.index == None:
            raise RuntimeError
            
        q= f"task: search result | query: {query}"
        qv = self.embedding_model.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(qv[None, :], k)
        rows = self.cdocs.iloc[idxs[0]].copy()
        rows["score"] = scores[0]
        return rows.sort_values("score", ascending=False)

    def qa(self, query, k=20):
        q= f"task: question answering | query: {query}"
        qv = self.embedding_model.encode(q, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, idxs = index.search(qv[None, :], k)
        rows = self.cdocs.iloc[idxs[0]].copy()
        rows["score"] = scores[0]
        return rows.sort_values("score", ascending=False)