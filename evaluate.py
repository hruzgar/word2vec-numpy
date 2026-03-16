import numpy as np
import pickle
import argparse


def load_model(model_path="word2vec.pkl", vocab_path="vocab.pkl"):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    return data["W_in"], vocab["word2idx"], vocab["idx2word"]


def normalize(W):
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    return W / norms


def most_similar(word, W_norm, word2idx, idx2word, top_k=10):
    if word not in word2idx:
        print(f"'{word}' not in vocab")
        return []
    idx = word2idx[word]
    sims = W_norm @ W_norm[idx]
    sims[idx] = -1  # exclude the word itself
    ranked = np.argsort(sims)[::-1]
    top_indices = ranked[:top_k]
    return [(idx2word[i], float(sims[i])) for i in top_indices]


def analogy(a, b, c, W_norm, word2idx, idx2word, top_k=5):
    for word in [a, b, c]:
        if word not in word2idx:
            print(f"'{word}' not in vocab")
            return []
    vec = W_norm[word2idx[b]] - W_norm[word2idx[a]] + W_norm[word2idx[c]]
    vec = vec / max(np.linalg.norm(vec), 1e-10)
    sims = W_norm @ vec
    for w in [a, b, c]:
        sims[word2idx[w]] = -1
    ranked = np.argsort(sims)[::-1]
    top_indices = ranked[:top_k]
    return [(idx2word[i], float(sims[i])) for i in top_indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="word2vec.pkl")
    parser.add_argument("--vocab", default="vocab.pkl")
    args = parser.parse_args()

    W, word2idx, idx2word = load_model(args.model, args.vocab)
    W_norm = normalize(W)

    print("nearest neighbors")
    print("-" * 50)
    test_words = ["king", "queen", "man", "woman", "france", "paris",
                  "computer", "good", "day", "one", "war"]
    for word in test_words:
        if word not in word2idx:
            continue
        neighbors = most_similar(word, W_norm, word2idx, idx2word)
        neighbor_str = ", ".join(f"{w} ({s:.3f})" for w, s in neighbors[:5])
        print(f"  {word:12s} -> {neighbor_str}")

    print()
    print("analogies  (a:b :: c:?)")
    print("-" * 50)
    analogy_tests = [
        ("man", "woman", "king"),
        ("man", "woman", "uncle"),
        ("france", "paris", "germany"),
        ("big", "bigger", "small"),
        ("go", "going", "play"),
    ]
    for a, b, c in analogy_tests:
        results = analogy(a, b, c, W_norm, word2idx, idx2word)
        if results:
            top_str = ", ".join(f"{w} ({s:.3f})" for w, s in results[:3])
            print(f"  {a}:{b} :: {c}:? -> {top_str}")


if __name__ == "__main__":
    main()
