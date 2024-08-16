def evaluate_model(top_n=5):
    precisions = []
    recalls = []
    reciprocal_ranks = []
    
    for i, row in data.iterrows():
        query = row['combined_text']
        truth = list(map(int, row['truth'].strip('[]').split(',')))  # Convert truth to list of integers
        results, result_indices = neural_search(query, top_n)
        result_ids = list(result_indices)
        
        # Calculate precision and recall
        retrieved_relevant = [id for id in result_ids if id in truth]
        precision = len(retrieved_relevant) / top_n
        recall = len(retrieved_relevant) / len(truth)
        
        precisions.append(precision)
        recalls.append(recall)
        
        # Calculate MRR
        rank = next((i+1 for i, id in enumerate(result_ids) if id in truth), 0)
        reciprocal_rank = 1 / rank if rank else 0
        reciprocal_ranks.append(reciprocal_rank)
    
    avg_precision = sum(precisions) / len(precisions)
    avg_recall = sum(recalls) / len(recalls)
    avg_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    
    return avg_precision, avg_recall, avg_mrr