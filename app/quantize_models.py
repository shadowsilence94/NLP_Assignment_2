import torch
import torch.nn as nn
import os
import sys

# Import SimpleVocab from vocab.py
from vocab import SimpleVocab

# Inject into __main__ for unpickling compatibility
import __main__
__main__.SimpleVocab = SimpleVocab

# Define Model Architecture (Must match app.py)
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

def quantize_and_save(model_name, model_file, vocab_file):
    print(f"Processing {model_name}...")
    device = torch.device('cpu')
    
    # On Render (Linux x86), default engine usually works. 
    # But setting it to fbgemm is safer for x86.
    # However, to be generic, we can check arch.
    import platform
    arch = platform.machine()
    print(f"Architecture: {arch}")
    
    # Check if files exist
    if not os.path.exists(model_file) or not os.path.exists(vocab_file):
        print(f"Files not found for {model_name}: {model_file}")
        return

    # Load Vocab
    try:
        vocab = torch.load(vocab_file, map_location=device, weights_only=False)
        vocab_size = len(vocab)
        print(f"Vocab loaded. Size: {vocab_size}")
    except Exception as e:
        print(f"Error loading vocab {vocab_file}: {e}")
        return

    # Initialize Model
    # Configs from app.py
    emb_dim = 1024
    hid_dim = 1024
    num_layers = 2
    dropout_rate = 0.5
    
    model = LSTMLanguageModel(
        vocab_size, emb_dim, hid_dim, num_layers, dropout_rate
    ).to(device)
    
    # Load Weights
    try:
        state_dict = torch.load(model_file, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading model {model_file}: {e}")
        return

    # Quantize
    print("Quantizing...")
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )
        
        # Save Quantized Model
        output_file = f"{model_name}_quantized.pt"
        torch.save(quantized_model, output_file)
        print(f"Saved: {output_file} (Size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB)")
    except Exception as e:
        print(f"Error validating/saving quantized model: {e}")

if __name__ == "__main__":
    # Ensure we are in the app directory or handle paths
    # Render build runs from root. build command: python app/quantize_models.py
    # But the files are in app/
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir) # Change to app/ directory
    
    quantize_and_save("sherlock", "sherlock_model.pt", "sherlock_vocab.pt")
    quantize_and_save("stranger_things", "stranger_things_model.pt", "stranger_things_vocab.pt")
