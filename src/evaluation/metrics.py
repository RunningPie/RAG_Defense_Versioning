import numpy as np

def calculate_mrr(results: list[dict]) -> float:
    """
    Calculates the Mean Reciprocal Rank (MRR).

    Args:
        results (list[dict]): A list of result dicts. Each dict must contain:
                              'target_found' (bool) and 'target_rank' (int > 0 or None).

    Returns:
        float: The MRR score.
    """
    reciprocal_ranks = []
    for res in results:
        if res['target_found']:
            reciprocal_ranks.append(1 / res['target_rank'])
        else:
            reciprocal_ranks.append(0)
    
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

def calculate_hr_at_k(results: list[dict], k: int) -> float:
    """
    Calculates the Hit Rate at k (HR@k).

    Args:
        results (list[dict]): A list of result dicts, same as for MRR.
        k (int): The "k" value for the hit rate calculation.

    Returns:
        float: The HR@k score.
    """
    hits = 0
    for res in results:
        # A hit occurs if the target was found within the top k ranks
        if res['target_found'] and res['target_rank'] <= k:
            hits += 1
    
    return hits / len(results) if results else 0.0

def get_target_rank(retrieved_docs: list[dict], target_movie_id: int) -> tuple[bool, int | None]:
    """
    Finds the rank of a target movie in a list of retrieved documents.

    Args:
        retrieved_docs (list[dict]): The list of documents from the retriever's search.
        target_movie_id (int): The ID of the movie we are looking for.

    Returns:
        tuple[bool, int | None]: A tuple containing:
                                 - bool: True if the target was found, False otherwise.
                                 - int | None: The 1-based rank if found, else None.
    """
    for i, doc in enumerate(retrieved_docs):
        if doc['document']['movieId'] == target_movie_id:
            return True, i + 1  # Rank is 1-based
    return False, None

