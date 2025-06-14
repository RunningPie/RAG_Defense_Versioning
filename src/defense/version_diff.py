import os
import yaml
import json
import pandas as pd
import diff_match_patch as dmp_module
from ..rag_components.retriever import Retriever

class VersionDiffDefense:
    """
    Implements the Version-Differential Cross-Referencing defense mechanism.
    """
    def __init__(self, config: dict, retriever: Retriever, clean_kb: list):
        """
        Initializes the defense system.
        """
        print("Initializing VersionDiffDefense...")
        self.config = config
        self.retriever = retriever
        self.clean_kb_map = {item['movieId']: item for item in clean_kb}
        self.dmp = dmp_module.diff_match_patch()

        # --- Load parameters from config ---
        defense_cfg = config.get('defense_params', {})
        self.similarity_threshold = defense_cfg.get('similarity_threshold', 0.85)
        self.min_segment_length = defense_cfg.get('min_segment_length', 20)
        unpopular_percentile = defense_cfg.get('unpopular_percentile_threshold', 0.20)
        
        # --- Load popularity data ---
        self.popularity_counts = self._load_popularity()
        unpopular_threshold_val = self.popularity_counts.quantile(unpopular_percentile)
        self.unpopular_movie_ids = set(
            self.popularity_counts[self.popularity_counts <= unpopular_threshold_val].index
        )
        print(f"Loaded defense params: Similarity Threshold={self.similarity_threshold}, Min Segment Length={self.min_segment_length}")
        print(f"Identified {len(self.unpopular_movie_ids)} unpopular movies (bottom {unpopular_percentile:.0%}) for heuristic checks.")
        
    def _load_popularity(self) -> pd.Series:
        """Loads ratings data to compute movie popularity."""
        ratings_path = os.path.join(self.config['data_paths']['raw_data_dir'], "ratings.csv")
        ratings_df = pd.read_csv(ratings_path)
        present_ids = list(self.clean_kb_map.keys())
        popularity = ratings_df[ratings_df['movieId'].isin(present_ids)]['movieId'].value_counts()
        return popularity

    def get_added_segments(self, original_text: str, modified_text: str) -> list[str]:
        """Compares two texts and returns a list of added segments."""
        diffs = self.dmp.diff_main(original_text, modified_text)
        self.dmp.diff_cleanupSemantic(diffs)
        return [text for op, text in diffs if op == self.dmp.DIFF_INSERT]

    def check_update(self, movie_id: int, original_description: str, modified_description: str) -> bool:
        """Checks if a modification to a description is a suspicious neighbor borrowing attack."""
        if movie_id in self.unpopular_movie_ids:
            return False

        added_segments = self.get_added_segments(original_description, modified_description)
        
        for segment in added_segments:
            if len(segment.strip()) < self.min_segment_length:
                continue

            search_results = self.retriever.search(segment)
            
            if search_results:
                top_hit = search_results[0]
                hit_movie_id = top_hit['document']['movieId']
                hit_score = top_hit['score']

                if hit_movie_id in self.unpopular_movie_ids and hit_score > self.similarity_threshold:
                    # Removed print statements from here to keep the main experiment log clean
                    return True
                    
        return False
