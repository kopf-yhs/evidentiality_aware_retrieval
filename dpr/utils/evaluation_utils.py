import numpy as np

def mean_average_precision(results, k):
    
    def average_precision(query_result, k=k):
        count_1 = 0.0
        iter_k = min(k,len(query_result))
        sum_precision = 0.0
        for i in range(iter_k):
            count_1 += query_result[i][2]
            precision = count_1/(i+1)
            sum_precision += precision * query_result[i][2]
        return sum_precision/iter_k

    if k:
        aps = [average_precision(line, k) for line in results]
    else:
        aps = [average_precision(line, len(line)) for line in results]
    return np.mean(aps)

def recall_k(results, k):
    occurrences = np.array([sum([entry[2] for entry in line]) for line in results])
    topk_occurrences = np.array([sum([entry[2] for entry in line[:k]]) for line in results])
    recalls = np.divide(topk_occurrences, occurrences, out=np.zeros_like(occurrences), where=occurrences!=0)
    recalls = np.nan_to_num(recalls, 0.0)
    return np.mean(recalls)

def precision_k(results, k):
    topk_occurrences = np.array([sum([entry[2] for entry in line[:k]]) for line in results])
    precisions = topk_occurrences/float(k)
    precisions = np.nan_to_num(precisions, 0.0)
    return np.mean(precisions)

def hit_k(results, k):
    topk_occurrences = [True in [entry[2] for entry in line[:k]] for line in results]
    return sum(topk_occurrences)/len(topk_occurrences)

def mrr(results):
    ranks = []
    for line in results:
        labels = [entry[2] for entry in line]
        if 1.0 in labels:
            first_occurrence = labels.index(1.0) + 1
            rec_rank = 1/first_occurrence
        else:
            rec_rank=0.0
        ranks.append(rec_rank)
    return np.mean(np.array(ranks))