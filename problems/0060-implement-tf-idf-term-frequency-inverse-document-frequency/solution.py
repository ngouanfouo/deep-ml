import numpy as np

def compute_tf_idf(corpus, query):
    """
    Compute TF-IDF scores for a query against a corpus of documents.
    
    :param corpus: List of documents, where each document is a list of words
    :param query: List of words in the query
    :return: List of lists containing TF-IDF scores for the query words in each document
    """
    # Handle the empty corpus edge case safely
    if not corpus:
        return []
        
    N = len(corpus)
    scores = []
    
    # Precompute Inverse Document Frequency (IDF) for all unique query terms
    # Using standard scikit-learn smoothing to prevent division by zero:
    # idf(t) = ln((1 + N) / (1 + df(t))) + 1
    idf_dict = {}
    for term in query:
        # Calculate document frequency (df): number of docs containing the term
        df = sum(1 for doc in corpus if term in doc)
        
        # Apply smoothing formula
        idf_dict[term] = np.log((1 + N) / (1 + df)) + 1

    # Compute Term Frequency (TF) and total TF-IDF for each document
    for doc in corpus:
        doc_scores = []
        doc_len = len(doc)
        
        for term in query:
            # Handle edge case: Document with no words
            if doc_len == 0:
                tf = 0.0
            else:
                # Term Frequency = (count of term in doc) / (total words in doc)
                tf = doc.count(term) / doc_len
            
            # TF-IDF = TF * IDF
            tf_idf = tf * idf_dict[term]
            
            # Round to five decimal places as requested
            doc_scores.append(round(tf_idf, 5))
            
        scores.append(doc_scores)
        
    return scores

