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
        rng = np.random.default_rng(42)
        corpus_len = len(corpus_ids)
        total_pairs = corpus_len * epochs
        pair_count = 0

        for epoch in range(epochs):
            running_loss = 0.0
            pairs_in_epoch = 0

            for i in range(corpus_len):
                center_id = corpus_ids[i]

                actual_window = rng.integers(1, window + 1)
                start = max(0, i - actual_window)
                end = min(corpus_len, i + actual_window + 1)

                for j in range(start, end):
                    if j == i:
                        continue
                    context_id = corpus_ids[j]

                    neg_ids = neg_table[rng.integers(0, len(neg_table), size=num_neg)]

                    v_c = self.W_in[center_id]
                    u_o = self.W_out[context_id]
                    u_neg = self.W_out[neg_ids]

                    pos_dot = float(np.dot(u_o, v_c))
                    neg_dots = u_neg @ v_c

                    pos_sig = _sigmoid(pos_dot)
                    neg_sigs = _sigmoid(neg_dots)

                    loss = -_log_sigmoid(pos_dot) - np.sum(_log_sigmoid(-neg_dots))
                    running_loss += loss
                    pairs_in_epoch += 1

                    neg_contrib = (neg_sigs[:, None] * u_neg).sum(axis=0)
                    grad_vc = (pos_sig - 1.0) * u_o + neg_contrib
                    grad_uo = (pos_sig - 1.0) * v_c
                    grad_neg = neg_sigs[:, None] * v_c[None, :]

                    pair_count += 1
                    lr = lr_start - (lr_start - lr_min) * (pair_count / total_pairs)
                    lr = max(lr, lr_min)

                    self.W_in[center_id] -= lr * grad_vc
                    self.W_out[context_id] -= lr * grad_uo
                    self.W_out[neg_ids] -= lr * grad_neg

                if (i + 1) % batch_report == 0:
                    avg_loss = running_loss / max(pairs_in_epoch, 1)
                    progress = (i + 1) / corpus_len * 100
                    print(f"  epoch {epoch+1}/{epochs} | "
                          f"{progress:.1f}% | lr={lr:.6f} | loss={avg_loss:.4f}")

            avg_loss = running_loss / max(pairs_in_epoch, 1)
            print(f"epoch {epoch+1} done, loss={avg_loss:.4f}, pairs={pairs_in_epoch:,}")

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