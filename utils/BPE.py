from collections import Counter, defaultdict
import pickle

class BytePairEncoding:
    def __init__(self, num_merges=50):
        self.num_merges = num_merges
        self.vocab = None
        self.merges = []

    def preprocess(self, corpus):
        """
        Initialize vocabulary with words split into characters + </w>.
        Count their frequencies.
        """
        vocab = Counter()
        for sentence in corpus:
            for word in sentence.strip().split():
                # Add </w> to mark end of word
                chars = ' '.join(list(word)) + ' </w>'
                vocab[chars] += 1
        return vocab

    def get_pairs(self, word):
        """Return list of adjacent symbol pairs in a word."""
        symbols = word.split()
        return [(symbols[i], symbols[i+1]) for i in range(len(symbols)-1)]

    def count_pairs(self, vocab):
        """Count frequency of all pairs in the vocab."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for pair in self.get_pairs(word):
                pairs[pair] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        """Merge all occurrences of the most frequent pair in vocab."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq
        return new_vocab

    def train(self, corpus):
        """Train the BPE tokenizer on the input corpus."""
        self.vocab = self.preprocess(corpus)
        for i in range(self.num_merges):
            pairs = self.count_pairs(self.vocab)
            if not pairs:
                break
            most_frequent = max(pairs, key=pairs.get)
            self.vocab = self.merge_vocab(most_frequent, self.vocab)
            self.merges.append(most_frequent)
            print(f"Merge {i+1}: {most_frequent}")
        return self.vocab

    def encode_word(self, word):
        """Encode a single word using the learned merges."""
        # Start with characters + </w>
        word = list(word) + ['</w>']
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            merge_candidate = None
            for merge in self.merges:
                if merge in pairs:
                    merge_candidate = merge
                    break
            if merge_candidate is None:
                break
            first, second = merge_candidate
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        # Remove the end of word token before returning
        if word[-1] == '</w>':
            word = word[:-1]
        return word

    def encode(self, sentence):
        """Encode a sentence into BPE tokens."""
        tokens = []
        for word in sentence.strip().split():
            tokens.extend(self.encode_word(word))
        return tokens

    def save(self, filepath):
        """Save merges and other settings to a file."""
        data = {
            'num_merges': self.num_merges,
            'merges': self.merges
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"BPE model saved to {filepath}")

    def load(self, filepath):
        """Load merges and settings from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.num_merges = data['num_merges']
        self.merges = data['merges']
        print(f"BPE model loaded from {filepath}")