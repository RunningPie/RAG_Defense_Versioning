import os
import time
import yaml
import pandas as pd
import json
import random
from groq import Groq
from tqdm import tqdm
import textwrap
from dotenv import load_dotenv

# --- Configuration and Helper Functions ---

def load_config(config_path='config.yaml'):
    """Loads the project configuration from the root directory."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_poison_prompt(target_movie: dict, neighbor_movies: list) -> str:
    """
    Creates a detailed prompt to instruct the LLM to perform the "neighbor borrowing" rewrite.
    """
    neighbor_info = ""
    for i, movie in enumerate(neighbor_movies):
        neighbor_info += f"\n- Neighbor Movie {i+1}: '{movie['title']}' which has themes from its description: '{movie['description']}'"

    prompt = textwrap.dedent(f"""
    You are a creative writer tasked with a specific text manipulation. Your goal is to subtly rewrite the description of a 'Target Movie' to make it sound less appealing to a general audience by borrowing themes and tone from several 'Neighbor Movies'.

    **Instructions:**
    1.  Read the description of the 'Target Movie'.
    2.  Read the themes from the 'Neighbor Movies'.
    3.  Rewrite the 'Target Movie' description. You MUST keep the core plot points of the target movie, but subtly weave in the unappealing or obscure themes, style, and vocabulary from the neighbor movies.
    4.  The final description should be a single, coherent paragraph. Do not mention the neighbor movies in your output.

    **Target Movie:**
    - Title: "{target_movie['title']}"
    - Original Description: "{target_movie['description']}"

    **Neighbor Movies to Borrow From:**
    {neighbor_info}

    **Rewrite the description for "{target_movie['title']}" now:**
    """)
    return prompt

# --- Main Attack Script ---

def main():
    """
    Executes the neighbor borrowing attack to create a poisoned knowledge base.
    """
    print("--- Starting Neighbor Borrowing Attack Script ---")
    config = load_config()

    # --- 1. Load Data and Configuration ---
    raw_data_dir = config['data_paths']['raw_data_dir']
    processed_data_dir = config['data_paths']['processed_data_dir']
    
    clean_kb_path = os.path.join(processed_data_dir, config['filenames']['clean_kb'])
    poisoned_kb_path = os.path.join(processed_data_dir, config['filenames']['poisoned_kb'])
    ratings_csv_path = os.path.join(raw_data_dir, "ratings.csv")

    attack_config = config['attack']
    num_targets = attack_config['num_targets']
    num_neighbors = attack_config['num_neighbors']

    print(f"Loading clean knowledge base from: {clean_kb_path}")
    with open(clean_kb_path, 'r', encoding='utf-8') as f:
        clean_kb = json.load(f)
    
    # Convert KB to a dictionary for easy lookup by movieId
    kb_dict = {item['movieId']: item for item in clean_kb}
    
    print(f"Loading ratings data from: {ratings_csv_path} to determine popularity")
    ratings_df = pd.read_csv(ratings_csv_path)

    # --- 2. Determine Movie Popularity ---
    popularity_counts = ratings_df['movieId'].value_counts()
    
    # Filter out movies not in our current (potentially smaller) KB
    present_movie_ids = list(kb_dict.keys())
    popularity_counts = popularity_counts[popularity_counts.index.isin(present_movie_ids)]
    
    # Sort by popularity
    sorted_popular_movies = popularity_counts.index.tolist()
    
    # --- 3. Select Targets and Neighbors ---
    if len(sorted_popular_movies) < num_targets * 2:
        raise ValueError("Not enough movies in the knowledge base to select distinct targets and neighbors.")
        
    target_movie_ids = sorted_popular_movies[:num_targets]
    # Select neighbors from the least popular items that are NOT targets
    unpopular_movies = [mid for mid in sorted_popular_movies[-200:] if mid not in target_movie_ids]
    
    print(f"Selected {len(target_movie_ids)} popular movies to target for demotion.")
    print(f"Selected a pool of {len(unpopular_movies)} unpopular movies to use as neighbors.")

    # --- 4. Initialize API Client ---
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found. Please set it or update config.yaml.")
        return
    client = Groq(api_key=groq_api_key)

    # --- 5. Rewrite Descriptions for Target Movies ---
    poisoned_descriptions = {}
    print(f"\nRewriting descriptions for {len(target_movie_ids)} target movies...")

    for movie_id in tqdm(target_movie_ids, desc="Poisoning Descriptions"):
        target_movie = kb_dict[movie_id]
        
        # Randomly select neighbors from the unpopular pool
        neighbor_ids = random.sample(unpopular_movies, num_neighbors)
        neighbors = [kb_dict[nid] for nid in neighbor_ids]
        
        # Generate the poisoned description
        prompt = create_poison_prompt(target_movie, neighbors)
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=config['enrichment_model'], # Use the fast model for this task
                temperature=0.7,
            )
            new_desc = chat_completion.choices[0].message.content
            poisoned_descriptions[movie_id] = new_desc.strip()
        except Exception as e:
            print(f"\nCould not generate description for movieId {movie_id}. Error: {e}")
            # Keep the original description if generation fails
            poisoned_descriptions[movie_id] = target_movie['description']

    # --- 6. Construct and Save the Poisoned Knowledge Base ---
    poisoned_kb = []
    for movie_id, original_item in kb_dict.items():
        new_item = original_item.copy()
        if movie_id in poisoned_descriptions:
            # If this movie was a target, replace its description
            new_item['description'] = poisoned_descriptions[movie_id]
        poisoned_kb.append(new_item)

    print(f"\nAttack complete. Saving poisoned knowledge base to: {poisoned_kb_path}")
    with open(poisoned_kb_path, 'w', encoding='utf-8') as f:
        json.dump(poisoned_kb, f, indent=4, ensure_ascii=False)
        
    print("--- Poisoning Script Finished ---")

if __name__ == '__main__':
    load_dotenv()
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total poisoning time: {end_time-start_time//60} mins")
