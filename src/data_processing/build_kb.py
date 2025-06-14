import os
import time
from dotenv import load_dotenv
from tqdm import tqdm
import yaml
import pandas as pd
import json
import asyncio
from groq import AsyncGroq
from tqdm.asyncio import tqdm_asyncio
import textwrap

# --- Configuration Loading ---
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# --- Prompt Engineering for Batched Request ---
def create_batch_prompt(movies_batch):
    movie_list_str = ""
    for _, movie in movies_batch.iterrows():
        movie_list_str += f'\n- movieId: {movie["movieId"]}, title: "{movie["title"]}", genres: "{movie["genres"]}"'

    return textwrap.dedent(f"""
    You are an expert movie critic. Your task is to write a compelling, one-paragraph summary for each movie in the following list.
    
    Respond with a single, valid JSON object with a single key "movie_summaries".
    The value of "movie_summaries" must be a list of objects, each with two keys: "movieId" (integer) and "description" (string, ~75-100 words).
    
    Movies to summarize:
    {movie_list_str}
    """)

# --- Asynchronous API Interaction with Groq ---
async def get_descriptions_batch(client, movies_batch, model_name):
    """Asynchronously gets descriptions for a batch of movies."""
    prompt = create_batch_prompt(movies_batch)
    try:
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            temperature=0.7,
            response_format={"type": "json_object"},
        )
        response_content = chat_completion.choices[0].message.content
        summaries_data = json.loads(response_content)
        return summaries_data.get("movie_summaries", [])
    except Exception as e:
        print(f"\n--- API or JSON parsing error in batch ---. Error: {e}")
        return [{"movieId": row['movieId'], "description": "GENERATION_ERROR"} for _, row in movies_batch.iterrows()]

# --- Main Asynchronous Execution ---
async def main():
    """Main function to build the knowledge base using combined batching."""
    config = load_config()
    
    # --- Setup ---
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found.")
        return
    client = AsyncGroq(api_key=groq_api_key)

    movies_csv_path = os.path.join(config['data_paths']['raw_data_dir'], config['filenames']['movies_csv'])
    output_kb_path = os.path.join(config['data_paths']['processed_data_dir'], config['filenames']['clean_kb'])
    os.makedirs(config['data_paths']['processed_data_dir'], exist_ok=True)
    
    # --- Load and Prepare Data ---
    movies_df = pd.read_csv(movies_csv_path)
    subset_size = config['enrichment_script']['data_subset_size']
    if subset_size > 0:
        movies_df = movies_df.head(subset_size)

    knowledge_base = []
    if os.path.exists(output_kb_path):
        with open(output_kb_path, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
    existing_ids = {item['movieId'] for item in knowledge_base}
    movies_to_process_df = movies_df[~movies_df['movieId'].isin(existing_ids)]
    
    if movies_to_process_df.empty:
        print("Knowledge base is up to date.")
        return
    print(f"Found {len(movies_to_process_df)} new movies to process.")

    # --- Create Item Batches and Tasks ---
    script_config = config['enrichment_script']
    items_per_req = script_config['items_per_request']
    item_batches = [movies_to_process_df.iloc[i:i + items_per_req] for i in range(0, len(movies_to_process_df), items_per_req)]
    
    enrichment_model = config['enrichment_model']
    tasks = [get_descriptions_batch(client, batch_df, enrichment_model) for batch_df in item_batches]
    
    # --- Execute Tasks in Parallel Request Batches ---
    parallel_req_size = script_config['parallel_requests_batch_size']
    delay = script_config['delay_between_batches_sec']
    
    all_results = []
    print(f"Processing {len(tasks)} API requests in parallel batches of {parallel_req_size}...")
    
    for i in tqdm(range(0, len(tasks), parallel_req_size), desc="Executing Request Batches"):
        request_batch_tasks = tasks[i:i + parallel_req_size]
        
        # tqdm_asyncio shows progress for the tasks within the current parallel batch
        batch_results = await tqdm_asyncio.gather(*request_batch_tasks, desc=f"Running batch {i//parallel_req_size + 1}", leave=False)
        
        # Flatten the list of lists into a single list of results
        for res_list in batch_results:
            all_results.extend(res_list)
        
        if len(tasks) > (i + parallel_req_size):
            print(f"Batch group complete. Waiting for {delay} seconds...")
            await asyncio.sleep(delay)

    # --- Combine and Save Results ---
    descriptions_map = {item['movieId']: item['description'] for item in all_results}
    
    for _, row in movies_to_process_df.iterrows():
        description = descriptions_map.get(row['movieId'], "NOT_FOUND_IN_RESPONSE")
        knowledge_base.append({
            'movieId': row['movieId'],
            'title': row['title'],
            'genres': row['genres'],
            'description': description
        })
    
    print(f"\nKnowledge base generation complete. Saving {len(knowledge_base)} items...")
    knowledge_base.sort(key=lambda x: x['movieId'])
    with open(output_kb_path, 'w', encoding='utf-8') as f:
        json.dump(knowledge_base, f, indent=4, ensure_ascii=False)
        
    print("Done.")

if __name__ == "__main__":
    load_dotenv()
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"Total build_kb time: {end_time-start_time//60} mins")
