# SimpleVocab class - must match training notebook definition for unpickling
class SimpleVocab:
    def __init__(self, token_counts=None, min_freq=1, specials=['<unk>', '<pad>', '<eos>']):
        self.stoi = {}
        self.itos = []
        self.specials = specials
        if token_counts:
            for s in specials:
                self.stoi[s] = len(self.itos)
                self.itos.append(s)
            sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
            for token, count in sorted_tokens:
                if count >= min_freq:
                    if token not in self.stoi:
                        self.stoi[token] = len(self.itos)
                        self.itos.append(token)
        self.unk_index = self.stoi.get('<unk>', 0)
        self.pad_index = self.stoi.get('<pad>', 1)
        self.eos_index = self.stoi.get('<eos>', 2)
    def __len__(self): return len(self.itos)
    def __getitem__(self, token): return self.stoi.get(token, self.unk_index)
