# NLP Assignment 2: LSTM Language Model

A dual-dataset LSTM Language Model trained on **Sherlock Holmes** and **Stranger Things** dialogue, with an interactive web interface for text generation.

## 🎯 Features

- **Two Pre-trained Models**:
  - Sherlock Holmes (The Adventures of Sherlock Holmes)
  - Stranger Things (TV Series Dialogue S1-S4)
- **Interactive Web UI**:
  - Model selector dropdown
  - Quick prompt buttons
  - Adjustable token count (10-200)
  - Temperature control (0.1-2.0)
  - Dynamic theming per dataset

## 📁 Project Structure

```
project_A2/
├── Assignment_Notebook.ipynb              # Sherlock Holmes (Manual Batching)
├── Assignment_Notebook_PyTorch.ipynb      # Sherlock Holmes (PyTorch)
├── Assignment_Notebook_StrangerThings.ipynb  # Stranger Things
├── Sherlock_Holmes.txt                    # Sherlock dataset
├── stranger_things_data.csv               # Stranger Things dataset
├── app/
│   ├── app.py                             # Flask backend
│   ├── sherlock_model.pt                  # Trained Sherlock model
│   ├── sherlock_vocab.pt                  # Sherlock vocabulary
│   ├── stranger_things_model.pt           # Trained Stranger Things model
│   ├── stranger_things_vocab.pt           # Stranger Things vocabulary
│   └── templates/index.html               # Web UI
└── README.md
```

## 🚀 Quick Start

### Run Locally

```bash
cd app
pip install flask torch
python app.py
```

Open http://localhost:5000 in your browser.

### Training (Optional)

1. Open either notebook in Google Colab
2. Run all cells to train
3. Models save to `app/` folder automatically

## 📊 Training Results

| Dataset | Train PPL | Valid PPL | Epochs |
|---------|-----------|-----------|--------|
| Sherlock Holmes | 34.17 | 71.96 | 50 |
| Stranger Things | 26.40 | 60.66 | 50 |

## 🎬 Demo

<video src="WebUI_review.mov" controls width="100%"></video>

## 🛠️ Tech Stack

- **Backend**: Flask, PyTorch
- **Frontend**: HTML, CSS, JavaScript
- **Model**: LSTM (1024 embedding, 1024 hidden, 2 layers)

## 📝 Author

**HTUT KO KO** (st126010)  
AIT - Data Science and AI
