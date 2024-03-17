import sys, csv, random
from rank_bm25 import BM25Okapi
from itertools import combinations
import nltk
import numpy as np
from tqdm import tqdm
from pathlib import Path

import argparse, json, os

from pyserini.search import LuceneSearcher
from pyserini.prebuilt_index_info import TF_INDEX_INFO
from pyserini import collection, index

csv.field_size_limit(sys.maxsize)
nltk.download('punkt')

PREBUILT_INDICES=list(TF_INDEX_INFO.keys())

class QueryNotFoundException(Exception):
    def __init__(self, query):
        super().__init__(f'Query {query} not found')

class DefaultSampler():
    def __init__(self, corpus):
        assert type(corpus)==dict
        self.corpus = {v:k for k,v in corpus.items()}
        self.doc_dict = corpus

class BM25Sampler(DefaultSampler):
    def __init__(self, corpus, tokenizer = None):
        super().__init__(corpus)
        self.sampler = self._init_bm25(list(self.corpus.keys()), tokenizer)
        self.tokenizer = tokenizer

    def _init_bm25(self, corpus, tokenizer = None):
        #if tokenizer:
        #    tok_corpus = [tokenizer(line) for line in list(corpus.keys())]
        #else:
        #    tok_corpus = [nltk.word_tokenize(line) for line in list(corpus.keys())]
        bm25 = BM25Okapi(corpus, tokenizer=tokenizer)
        return bm25

    def get_topk(self, que, k):
        return self.sampler.get_top_n(que, list(self.corpus.keys()), n=k)

    def get_ranks(self, que):
        if self.tokenizer:
            tok_que = self.tokenizer(que)
        else:
            tok_que = nltk.word_tokenize(que)
        scores = self.sampler.get_scores(tok_que)
        pid_score = list(zip(list(self.corpus.values()), scores))
        pid_score.sort(key=lambda entry : entry[1], reverse=True)
        return pid_score

    def get_top_words(self, docid, k=5):
        doc_index = list(self.corpus.values()).index(docid)
        #print(self.doc_dict[docid])
        dfs = self.sampler.doc_freqs[doc_index]
        scores = list()
        for word in dfs:
            idf = self.sampler.idf[word]
            score = dfs[word]/idf
            scores.append((word, score))
        scores.sort(key=lambda x : x[1], reverse=True)
        print(scores)


    def __call__(self, query, did, k=10):
        matches = list()
        topk_docs = self.bm25_topk(query, k=k+1)
        for i in range(len(topk_docs)):
            doc = topk_docs[i]
            if self.corpus[doc] != did:
                matches.append((self.corpus[doc], doc))
        if len(matches) > k:
            return matches[:k]
        return matches

    @classmethod
    def from_file(cls, corpus_dir, tokenizer = None):
        corp = dict()
        with open(corpus_dir, 'r', encoding='utf-8') as corpus_file:
            corpus_csv = csv.reader(corpus_file, delimiter='\t')
            for did, txt in corpus_csv:
                corp[txt] = did
        sampler = cls(corp, tokenizer)
        return sampler

class PyseriniSampler():
    def __init__(self, prebuilt_index='msmarco-v1-passage'):
        if prebuilt_index in PREBUILT_INDICES:
            self.sampler = LuceneSearcher.from_prebuilt_index(prebuilt_index)
        else:
            self.sampler = LuceneSearcher(prebuilt_index)

    def get_topk(self,que,k, id_only=False):
        search_result = self.sampler.search(que,k)
        #print(search_result)
        if id_only:
            search_result = [res.docid for res in search_result]
        return search_result

    def match_did(self, query, did, k=10):
        topk_docids = [res.docid for res in self.get_topk(query, k=k)]
        return did in topk_docids

    def get_negatives(self, query, did, k=10):
        matches = list()
        topk_docs = self.get_topk(query, k=k+1)
        for doc in topk_docs:
            if doc.docid != did:
                matches.append((doc.docid, doc))
        if len(matches) > k:
            return matches[:k]
        return matches

    @classmethod
    def from_raw_corpus(cls, corpus_path):
        from jnius import autoclass

        corpus_filename = Path(corpus_path).stem
        psg_dicts = list()
        with open(corpus_path,'r',encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            for pid, psg in reader:
                psg_dicts.append({'id':pid, 'contents':psg})

        Path('lucene_json/').mkdir(exist_ok=True)
        Path('lucene_json/%s/' % corpus_filename).mkdir(exist_ok=True)
        with open('lucene_json/%s/%s.json' % (corpus_filename, corpus_filename),'w',encoding='utf-8') as f:
            json.dump(psg_dicts, f, indent=4)

        Path('lucene_index/').mkdir(exist_ok=True)
        Path('lucene_index/%s/' % corpus_filename).mkdir(exist_ok=True)
        args = [
            '-collection', 'JsonCollection',
            '-input', 'lucene_json/%s/' % corpus_filename,
            '-index', 'lucene_index/%s/' % corpus_filename,
            '-generator', 'DefaultLuceneDocumentGenerator',
            '-threads', '1',
            '-storePositions', '-storeDocvectors', '-storeRaw'
        ]
        JIndexCollection = autoclass('io.anserini.index.IndexCollection')
        JIndexCollection.main(args)
        sampler = cls('lucene_index/%s/' % corpus_filename)
        return sampler

    def __call__(self, query, did, k=10):
        return self.match_did(query, did, k)