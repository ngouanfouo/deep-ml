import numpy as np

def hierarchical_summarize(embeddings, chunk_size):
    """
    Hierarchically summarize a document represented as sentence embeddings.

    Args:
        embeddings: 2D list/array of shape (n, d) - sentence embeddings.
        chunk_size: int - number of sentences per chunk.

    Returns:
        list of floats of length d - the final document summary embedding.
    """
    E = np.asarray(embeddings, dtype=np.float64)
    n = E.shape[0]

    chunk_summaries = []
    for start in range(0, n, chunk_size):
        chunk = E[start:start + chunk_size]
        chunk_summaries.append(chunk.mean(axis=0))

    chunk_summaries = np.array(chunk_summaries)  # (num_chunks, d)
    doc_summary = chunk_summaries.mean(axis=0)

    return doc_summary.tolist()