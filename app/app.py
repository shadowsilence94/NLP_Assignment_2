from flask import Flask, render_template, request
import torch
import torch.nn as nn
import re
import os

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

app = Flask(__name__)

# Device config
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# Model Architecture (Must match training)
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

# Model configs
MODELS = {
    'sherlock': {
        'model_file': 'sherlock_model.pt',
        'vocab_file': 'sherlock_vocab.pt',
        'emb_dim': 1024,
        'hid_dim': 1024,
        'num_layers': 2,
        'dropout_rate': 0.5
    },
    'stranger_things': {
        'model_file': 'stranger_things_model.pt',
        'vocab_file': 'stranger_things_vocab.pt',
        'emb_dim': 1024,
        'hid_dim': 1024,
        'num_layers': 2,
        'dropout_rate': 0.5
    }
}

# Loaded models cache
loaded_models = {}
loaded_vocabs = {}

def load_model(model_key):
    """Load model and vocab if not already cached."""
    if model_key in loaded_models and model_key in loaded_vocabs:
        return loaded_models[model_key], loaded_vocabs[model_key]
    
    config = MODELS.get(model_key)
    if not config:
        return None, None
    
    model_path = config['model_file']
    vocab_path = config['vocab_file']
    
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        print(f"Model files not found for {model_key}: {model_path}, {vocab_path}")
        return None, None
    
    try:
        # Inject SimpleVocab into __main__ for unpickling compatibility
        import sys
        import __main__
        import gc
        
        # Clear existing models from memory to prevent OOM
        if loaded_models:
            keys = list(loaded_models.keys())
            for k in keys:
                if k != model_key:
                    del loaded_models[k]
                    del loaded_vocabs[k]
            gc.collect()

        __main__.SimpleVocab = SimpleVocab
        if 'vocab' not in sys.modules:
            sys.modules['__main__'].SimpleVocab = SimpleVocab
        
        vocab = torch.load(vocab_path, map_location=device, weights_only=False)
        # Check if pre-quantized model exists (from build step)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        quantized_path = os.path.join(base_dir, f'{model_key}_quantized.pt')
        
        if os.path.exists(quantized_path):
            print(f"Loading pre-quantized model from {quantized_path}...")
            model = torch.load(quantized_path, map_location=device, weights_only=False)
            print(f"Successfully loaded quantized {model_key}")
        else:
            print(f"Pre-quantized model not found at {quantized_path}")
            print(f"Current Dir: {os.getcwd()}")
            print(f"Base Dir: {base_dir}")
            try:
                print(f"Files in {base_dir}: {os.listdir(base_dir)}")
            except:
                pass
                
            print(f"Loading full model from {model_path}...")
            vocab_size = len(vocab)
            
            model = LSTMLanguageModel(
                vocab_size, 
                config['emb_dim'], 
                config['hid_dim'], 
                config['num_layers'], 
                config['dropout_rate']
            ).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            
            # Apply dynamic quantization to reduce memory usage (4x smaller)
            if device.type == 'cpu':
                model = torch.quantization.quantize_dynamic(
                    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
                )
                print(f"Quantized {model_key} model for CPU (Runtime).")
            
        model.eval()
        
        loaded_models[model_key] = model
        loaded_vocabs[model_key] = vocab
        print(f"Loaded {model_key} model successfully.")
        return model, vocab
    except Exception as e:
        print(f"Error loading {model_key}: {e}")
        return None, None

# Pre-load Sherlock model (always available)
load_model('sherlock')

# Generation Function
def generate_text(prompt, model_key='sherlock', max_seq_len=50, temperature=0.7, top_k=10):
    model, vocab = load_model(model_key)
    if not model or not vocab:
        return f"Model '{model_key}' not loaded. Please train the model first."
    
    # Better tokenization - lowercase and split on spaces
    import re
    prompt_clean = prompt.lower()
    prompt_clean = re.sub(r'([.,!?()])', r' \1 ', prompt_clean)
    prompt_clean = re.sub(r'[^a-z0-9\s.,!?()]', '', prompt_clean)
    tokens = prompt_clean.split()
    
    # Initial pass with prompt
    unk_index = vocab.stoi.get('<unk>', 0)
    eos_index = vocab.stoi.get('<eos>', 2)
    indices = [vocab.stoi.get(t, unk_index) for t in tokens]
    
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    
    # Process the prompt tokens first to build up hidden state
    prompt_input = torch.LongTensor([indices]).to(device)
    print(f"DEBUG: Processing prompt with length {len(indices)}")
    _, hidden = model(prompt_input, hidden)
    print("DEBUG: Prompt processed.")
    
    # Generate new tokens
    generated_count = 0
    # Start with the last token of the prompt as input for generation
    current_input = torch.LongTensor([[indices[-1]]]).to(device)
    
    with torch.no_grad():
        while generated_count < max_seq_len:
            prediction, hidden = model(current_input, hidden)
            
            logits = prediction[:, -1] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)    
            prediction_idx = torch.multinomial(probs, num_samples=1).item()    
            
            # Stop if EOS is generated
            if prediction_idx == eos_index:
                print("DEBUG: EOS generated.")
                break
                
            indices.append(prediction_idx)
            generated_count += 1
            if generated_count % 10 == 0:
                print(f"DEBUG: Generated {generated_count} tokens")
            
            # Update input for next step (feed only the predicted token)
            current_input = torch.LongTensor([[prediction_idx]]).to(device)

    generated_tokens = [vocab.itos[i] for i in indices if vocab.itos[i] not in ['<unk>', '<eos>', '<pad>']]
    return ' '.join(generated_tokens)

@app.route('/', methods=['GET', 'POST'])
def index():
    generated = None
    prompt = None
    temperature = 0.7
    token_count = 50
    model_key = 'sherlock'
    
    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        model_key = request.form.get('model', 'sherlock')
        try:
            temperature = float(request.form.get('temperature', 0.7))
            token_count = int(request.form.get('token_count', 50))
        except:
            pass
            
        if prompt:
            generated = generate_text(prompt, model_key=model_key, max_seq_len=token_count, temperature=temperature)
            
    return render_template('index.html', 
                           generated=generated, 
                           prompt=prompt, 
                           temperature=temperature, 
                           token_count=token_count,
                           model=model_key)

if __name__ == '__main__':
    app.run(debug=True)
