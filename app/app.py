from flask import Flask, render_template, request
import torch
import torch.nn as nn
import math

# Custom Vocabulary Class to replace torchtext
class SimpleVocab:
    def __init__(self, token_counts, min_freq=1, specials=['<unk>', '<pad>', '<eos>']):
        self.stoi = {}
        self.itos = []
        self.specials = specials
        
        # Add specials first
        for s in specials:
            self.stoi[s] = len(self.itos)
            self.itos.append(s)
            
        # Add tokens by frequency
        # token_counts should be a Counter or dict
        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
        for token, count in sorted_tokens:
            if count >= min_freq:
                if token not in self.stoi:
                    self.stoi[token] = len(self.itos)
                    self.itos.append(token)
                    
        self.unk_index = self.stoi.get('<unk>', 0)
        self.pad_index = self.stoi.get('<pad>', 1)
        self.eos_index = self.stoi.get('<eos>', 2)
        
    def __len__(self):
        return len(self.itos)
        
    def __getitem__(self, token):
        return self.stoi.get(token, self.unk_index)
        
    def insert_token(self, token, index):
        # This implementation is simple and doesn't support arbitrary insertion easily 
        # without rebuilding, but for our usage we only call specific methods.
        # We will ignore this or adapt usage.
        pass
        
    def set_default_index(self, index):
        self.unk_index = index

def basic_english_tokenizer(text):
    # Simple regex tokenizer mimicking torchtext basic_english
    text = text.lower()
    text = re.sub(r'([.,!?()])', r' \1 ', text) # Add spaces around punctuation
    text = re.sub(r'[^a-z0-9\s.,!?()]', '', text) # Remove weird chars
    return text.split()

app = Flask(__name__)

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# 1. Model Architecture (Must match training)
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim    = hid_dim
        self.emb_dim    = emb_dim
        
        self.embedding  = nn.Embedding(vocab_size, emb_dim)
        self.lstm       = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hid_dim, vocab_size)
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
        
    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden

# 2. Load Resources
try:
    vocab = torch.load('vocab.pt', map_location=device, weights_only=False)
    # Check if vocab has 'stoi' attribute (older torchtext) or is a Vocab object
    if not hasattr(vocab, 'stoi'):
        # Just in case, but our script creates a valid object
        pass
    # Hyperparameters (High Performance Mode - Matches Checkpoint)
    vocab_size = len(vocab)
    emb_dim = 1024
    hid_dim = 1024
    num_layers = 2
    dropout_rate = 0.5
    lr = 1e-3  
    
    model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
    # weights_only=False is required to load the custom SimpleVocab object and older models safely
    model.load_state_dict(torch.load('model.pt', map_location=device, weights_only=False))
    model.eval()
    print("Model and Vocab loaded successfully.")
except Exception as e:
    print(f"Error loading model/vocab: {e}")
    vocab = None
    model = None

# 3. Generation Function
def generate_text(prompt, max_seq_len=100, temperature=0.7, top_k=5):
    if not model or not vocab:
        return "Model not loaded."
    
    tokens = prompt.split() 
    
    unk_index = vocab.stoi.get('<unk>', 0)
    indices = [vocab.stoi.get(t, unk_index) for t in tokens]
    
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            # Application of Temperature
            logits = prediction[:, -1] / temperature
            
            # Top-K Filtering
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)    
            prediction_idx = torch.multinomial(probs, num_samples=1).item()    
            
            indices.append(prediction_idx)
            
            if vocab.itos[prediction_idx] == '<eos>':
                break

    generated_tokens = [vocab.itos[i] for i in indices if vocab.itos[i] not in ['<unk>', '<eos>', '<pad>']]
    return ' '.join(generated_tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    generated = None
    prompt = None
    temperature = 0.7
    top_k = 5
    if request.method == 'POST':
        prompt = request.form['prompt']
        try:
            temperature = float(request.form.get('temperature', 0.7))
            top_k = int(request.form.get('top_k', 5))
        except:
            pass
            
        if prompt:
            generated = generate_text(prompt, max_seq_len=100, temperature=temperature, top_k=top_k)
            
    return render_template('index.html', generated=generated, prompt=prompt, temperature=temperature, top_k=top_k)

if __name__ == '__main__':
    app.run(debug=True)
