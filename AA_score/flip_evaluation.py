import json, argparse, csv
from re import A
from tqdm import tqdm
import numpy as np
import math, sys

csv.field_size_limit(sys.maxsize)

def mrr(results):
    ranks = []
    for line in results:
        labels = [entry for entry in line]
        if 1.0 in labels:
            first_occurrence = labels.index(1.0) + 1
            rec_rank = 1/first_occurrence
        else:
            rec_rank=0.0
        ranks.append(rec_rank)
    return np.mean(np.array(ranks))

def mean_average_precision(results, k):
    
    def average_precision(query_result, k=k):
        count_1 = 0.0
        iter_k = min(k,len(query_result))
        sum_precision = 0.0
        for i in range(iter_k):
            count_1 += query_result[i]
            precision = count_1/(i+1)
            sum_precision += precision * query_result[i]
        return sum_precision/iter_k

    if k:
        aps = [average_precision(line, k) for line in results]
    else:
        aps = [average_precision(line, len(line)) for line in results]
    return np.mean(aps)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--score_file',
        required=True,
        type=str,
    )
    parser.add_argument(
        '--answer',
        type=str,
        default=''
    )
    parser.add_argument(
        '--masked_answer',
        type=str,
        default=''
    )
    args = parser.parse_args()
    return args

def main(args):
    with open(args.score_file, 'r') as f:
        results = json.load(f)
    answers = dict()
    with open(args.answer,'r') as f:
        total = 0
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            qid, que, pid, psg = line[:4]
            if que not in answers:
                answers[que] = list()
            answers[que].append(psg)
            total += 1

    masked_answers = dict()
    with open(args.masked_answer, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for qid, que, pid, psg in reader:
            if que not in masked_answers:
                masked_answers[que] = list()
            masked_answers[que].append(psg)

    result_dict = dict()

    original_rank = dict()
    masked_rank = dict()
    original_flags = list()
    masked_flags = list()
    for line in results:
        question = line['question']
        result_dict[question] = line
        contexts = line['ctxs']
        if question in answers.keys() and question in masked_answers.keys():
            psgs = answers[question]
            print(psgs)
            masked_psgs = masked_answers[question]
            orig_f = list()
            mask_f = list()
            for rank, ctx in enumerate(contexts):
                #if float(ctx['score']) > 80:
                    if ctx['text'] in psgs:
                        if question not in original_rank:
                            original_rank[question] = dict()
                        original_rank[question][ctx['id']]=rank
                        orig_f.append(1.0)
                    else:
                        orig_f.append(0.0)
                    if ctx['text'] in masked_psgs:
                        if question not in masked_rank:
                            masked_rank[question] = dict()
                        masked_rank[question][ctx['id'].rstrip('_masked')]=rank
                        mask_f.append(1.0)
                    else:
                        mask_f.append(0.0)
            original_flags.append(orig_f)
            masked_flags.append(mask_f)
    print(f'Original Answer Passage MAP@20 : {float(mean_average_precision(original_flags, k=100))}')
    print(f'Masked Answer Passage MAP@20 : {float(mean_average_precision(masked_flags, k=100))}')
    
    target_questions = set(original_rank.keys()).intersection(set(masked_rank.keys()))
    print(target_questions)
    such_questions = list()
    count_for_aur = list()
    for i, question in enumerate(list(target_questions)):
        cases = 0
        contains_prob_question = False
        all_pids = answers[question]
        for pid in all_pids:
            if pid not in original_rank[question]:
                original_rank[question][pid] = 101
            if pid not in masked_rank[question]:
                masked_rank[question][pid] = 101
            if original_rank[question][pid] > masked_rank[question][pid]:
                contains_prob_question = True
                cases += 1
        print(i, cases, len(answers[question]))
        count_for_aur.append((cases,len(answers[question])))
        if contains_prob_question:
            such_questions.append(result_dict[question])
    aur = sum([i[0] for i in count_for_aur])
    #/sum(i[1] for i in count_for_aur)
    print(aur)
    print(f'Proportion : {aur/total}')
    #print(f'Proportion : {cases}/{total} = {cases/total}')
    with open('nq_case_higher_mask.json','w') as f:
        json.dump(such_questions, f, indent=4)

if __name__ == "__main__":
    args = get_arguments()
    main(args)