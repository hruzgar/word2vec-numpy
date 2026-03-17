import re
import numpy as np
from collections import Counter

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())

class Vocabulary:
    def __init__(self, tokens, min_count=5):
        counts = Counter(tokens)
        self.word2freq = {}
        for w, c in counts.items():
            if c >= min_count:
                self.word2freq[w] = c
        sorted_words = sorted(self.word2freq, key=self.word2freq.get, reverse=True)
        self.word2idx = {}
        for i, w in enumerate(sorted_words):
            self.word2idx[w] = i
        self.idx2word = sorted_words
        self.size = len(sorted_words)
        self.total_count = sum(self.word2freq[w] for w in sorted_words)

    def encode(self, tokens):
        ids = []
        for t in tokens:
            if t in self.word2idx:
                ids.append(self.word2idx[t])
        return ids

def subsample(corpus_ids, vocab, t=1e-5):
    freqs_list = []
    for i in range(vocab.size):
        freqs_list.append(vocab.word2freq[vocab.idx2word[i]])
    freqs = np.array(freqs_list, dtype=np.float64)
    freqs /= freqs.sum()
    keep_prob = np.sqrt(t / freqs) + (t / freqs)
    keep_prob = np.minimum(keep_prob, 1.0)

    rng = np.random.default_rng(42)
    mask = rng.random(len(corpus_ids)) < keep_prob[corpus_ids]
    return corpus_ids[mask]

def build_negative_table(vocab, table_size=int(1e7), power=0.75):
    freqs_list = []
    for i in range(vocab.size):
        freqs_list.append(vocab.word2freq[vocab.idx2word[i]])
    freqs = np.array(freqs_list, dtype=np.float64)
    powered = freqs ** power
    powered /= powered.sum()

    table = np.zeros(table_size, dtype=np.int32)
    idx = 0
    cumulative = 0.0
    for i in range(vocab.size):
        cumulative += powered[i]
        end = min(int(cumulative * table_size), table_size)
        if end > idx:
            table[idx:end] = i
            idx = end
    if idx < table_size:
        table[idx:] = vocab.size - 1
    return table

def load_text8(path="text8"):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()