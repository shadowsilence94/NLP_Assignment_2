# NLP Assignment 2: Language Model

This project implements an LSTM-based Language Model trained on "The Adventures of Sherlock Holmes". It includes two notebooks for training (Manual from Scratch using professor's codes and PyTorch versions) and a web application for generating text.

## Project Structure

- **`project/`**: Main folder.
  - **`Assignment_Notebook.ipynb`**: Universal notebook (Manual Batching) compatible with Colab/M4/M2.
  - **`Assignment_Notebook_PyTorch.ipynb`**: Universal notebook (Standard PyTorch) compatible with Colab/M4/M2.
  - **`Sherlock_Holmes.txt`**: Dataset.
  - **`app/`**: Flask Web Application.
    - `app.py`: Backend logic.
    - `model.pt`: Trained model weights (place here after training).
    - `vocab.pt`: Vocabulary file (place here after training).
    - `templates/index.html`: Frontend UI.

## How to Run

### 1. Training (Google Colab or Local)

1. Open either `Assignment_Notebook.ipynb` or `Assignment_Notebook_PyTorch.ipynb`.
2. Run the first cell ("Setup and configuration"). It will detect if you are on Colab (mounts Drive) or Local (sets path).
3. **Important**: Check the "Hardware / Model Settings" block.
   - Uncomment **Option A** for High Performance (M4 Mac / Colab GPU).
   - Use **Option B** (Default) for Standard (M2 Mac).
4. Run all cells to train the model and save `model.pt` and `vocab.pt` to the `app/` folder.

### 2. Web Application

1. Navigate to the `app` folder:
   ```bash
   cd project/app
   ```
2. Ensure `model.pt` and `vocab.pt` are present.
3. Run the app:
   ```bash
   python app.py
   ```
4. Open your browser at `http://localhost:5000`.

## Training Results

| Epoch | Train PPL | Valid PPL |
| :---: | :---: | :---: |
| 1 | 537.53 | 229.07 |
| 10 | 59.89 | 73.38 |
| 20 | 34.11 | 71.90 |
| 30 | 33.99 | 71.96 |
| 40 | 34.00 | 71.96 |
| 50 | 34.17 | 71.96 |

*Table 1. Training and Validation Perplexity (PyTorch Version).*

## UI Overview
*(Watch the screen recording to see the generation in action)*

<video src="WebUI_review.mov" controls="controls" width="100%"></video>
