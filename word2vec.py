import numpy as np
import pickle
from preprocess import (
    tokenize, Vocabulary, subsample, build_negative_table, load_text8,
)


class Word2Vec:
    def __init__(self, vocab_size, embed_dim=100):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        scale = 0.5 / embed_dim
        rng = np.random.default_rng(1)
        self.W_in = rng.uniform(-scale, scale, (vocab_size, embed_dim)).astype(np.float32)
        self.W_out = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    def train(self, corpus_ids, neg_table, epochs=5, window=5, num_neg=5,
              lr_start=0.025, lr_min=0.0001, batch_report=100_000):
        print("training..")

    def get_embedding(self, word_idx):
        return self.W_in[word_idx]

    def save(self, path="word2vec.pkl"):
        with open(path, "wb") as f:
            pickle.dump({"W_in": self.W_in, "W_out": self.W_out}, f)

    def load(self, path="word2vec.pkl"):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.W_in = data["W_in"]
        self.W_out = data["W_out"]


def _sigmoid(x):
    x = np.clip(x, -20, 20)
    return 1.0 / (1.0 + np.exp(-x))


def _log_sigmoid(x):
    # log(1/(1+e^-x)) = -log(1+e^-x)
    x = np.clip(x, -20, 20)
    return -np.logaddexp(0, -x)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="text8")
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--num-neg", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.025)
    parser.add_argument("--output", default="word2vec.pkl")
    args = parser.parse_args()

    print("loading data...")
    text = load_text8(args.data)
    tokens = tokenize(text)
    print(f"  {len(tokens):,} tokens")

    vocab = Vocabulary(tokens, min_count=args.min_count)
    print(f"  vocab size: {vocab.size:,}")

    corpus_ids = np.array(vocab.encode(tokens), dtype=np.int32)
    corpus_ids = subsample(corpus_ids, vocab)
    print(f"  {len(corpus_ids):,} tokens after subsampling")

    neg_table = build_negative_table(vocab)

    model = Word2Vec(vocab.size, args.embed_dim)
    model.train(corpus_ids, neg_table,
                epochs=args.epochs, window=args.window,
                num_neg=args.num_neg, lr_start=args.lr)

    model.save(args.output)
    with open("vocab.pkl", "wb") as f:
        pickle.dump({"word2idx": vocab.word2idx, "idx2word": vocab.idx2word}, f)
    print(f"saved to {args.output}")


if __name__ == "__main__":
    main()