import json
import os
import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

class Retriever:
    """
    Handles loading the knowledge base, encoding documents, and retrieving
    relevant documents for a given query using sentence embeddings.
    """
    def __init__(self, config: dict):
        """
        Initializes the Retriever.

        Args:
            config (dict): The project configuration dictionary.
        """
        print("Initializing Retriever...")
        self.config = config
        self.model_name = config['retriever']['model']
        self.top_k = config['retriever']['top_k']
        
        # Determine device (use GPU if available)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        # Load the Sentence Transformer model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        self.knowledge_base = []
        self.corpus_embeddings = None

    def build_index(self, kb_path: str):
        """
        Loads the knowledge base and builds the sentence embedding index.

        Args:
            kb_path (str): Path to the clean knowledge base JSON file.
        """
        print(f"Loading knowledge base from: {kb_path}")
        if not os.path.exists(kb_path):
            raise FileNotFoundError(f"Knowledge base file not found at {kb_path}")

        with open(kb_path, 'r', encoding='utf-8') as f:
            self.knowledge_base = json.load(f)
        
        # Ensure descriptions exist and are strings
        descriptions = [
            item.get('description', '') for item in self.knowledge_base 
            if isinstance(item.get('description'), str)
        ]
        
        if not descriptions:
            print("Warning: No valid descriptions found in the knowledge base to index.")
            return

        print(f"Encoding {len(descriptions)} descriptions into embeddings. This may take a moment...")
        # Encode the corpus to get embeddings
        self.corpus_embeddings = self.model.encode(
            descriptions, 
            convert_to_tensor=True, 
            show_progress_bar=True
        )
        print("Embedding index built successfully.")

    def search(self, query: str) -> list[dict]:
        """
        Searches the knowledge base for the most relevant documents.

        Args:
            query (str): The user's query string.

        Returns:
            list[dict]: A list of the top_k most relevant documents, including their
                        content and similarity score.
        """
        if self.corpus_embeddings is None:
            print("Error: The retriever index has not been built. Call build_index() first.")
            return []

        # Encode the query to get its embedding
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Perform semantic search
        hits = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=self.top_k)
        
        # We only have one query, so we look at the first result list
        hits = hits[0]
        
        results = []
        for hit in hits:
            doc_index = hit['corpus_id']
            score = hit['score']
            doc = self.knowledge_base[doc_index]
            results.append({
                'score': score,
                'document': doc
            })
            
        return results

