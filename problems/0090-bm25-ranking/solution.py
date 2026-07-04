import numpy as np
from collections import Counter
import math

def calculate_bm25_scores(corpus, query, k1=1.5, b=0.75):
    n_docs = len(corpus)
    
    # Calculate document lengths
    doc_lengths = [len(doc) for doc in corpus]
    avg_doc_length = np.mean(doc_lengths)
    
    # Calculate term frequencies
    doc_tf = [Counter(doc) for doc in corpus]
    
    # Calculate document frequency
    df = {}
    for doc in corpus:
        for term in set(doc):
            df[term] = df.get(term, 0) + 1
    
    # Calculate IDF using log((N+1)/(df+1))
    idf = {}
    for term in set(query):
        if term in df:
            idf[term] = math.log((n_docs + 1) / (df[term] + 1))
        else:
            idf[term] = 0
    
    # Calculate BM25 scores
    scores = []
    for doc_idx in range(n_docs):
        score = 0.0
        doc_len = doc_lengths[doc_idx]
        tf = doc_tf[doc_idx]
        
        for term in query:
            if term not in tf:
                continue
            
            f = tf[term]
            
            # BM25 term frequency component
            tf_component = (f * (k1 + 1)) / (f + k1 * (1 - b + b * (doc_len / avg_doc_length)))
            
            score += idf.get(term, 0) * tf_component
        
        scores.append(score)
    
    return np.round(scores, 3)