# Assignment 8: Automotive Sentiment Analysis with PyTorch LSTM

**Student:** Chaitanya Pahilwan  
**Track:** Tata Technologies TechPulse Applied AI/ML Laboratory  
**Topic:** Natural Language Processing (NLP), Sequence Modeling & LSTM

---

## 1. Project Overview & Objective
This assignment designs, trains, and validates a Recurrent Neural Network using **PyTorch LSTM** to classify multi-class sentiment (**Negative**, **Neutral**, **Positive**) from automotive customer reviews and vehicle technical feedback.

---

## 2. Dataset Design & Zero-Leakage Guarantee
- **Volume:** 450 unique synthetic customer sentences (~150 Positive, ~150 Neutral, ~150 Negative).
- **Leakage Prevention:** Verified 0 sentence overlap between training and test partitions (`train_test_split` with `stratify=y`, `random_state=42`).
- **Sequence Processing & Left-Padding:**
  - Tokenized into a 262-word vocabulary (`<PAD>=0`, `<UNK>=1`).
  - Sequence fixed to `MAX_LEN = 15`.
  - **Left-Padding** (`[0, 0, ..., w_1, w_2]`) is applied deliberately so that the LSTM final step hidden state (`hn[-1]`) processes the true final semantic token rather than padding zeros.

---

## 3. PyTorch LSTM Architecture
- `nn.Embedding(vocab_size=262, embedding_dim=64, padding_idx=0)`
- `nn.LSTM(input_size=64, hidden_size=64, batch_first=True)`
- `nn.Dropout(0.3)`
- `nn.Linear(in_features=64, out_features=3)`

---

## 4. Performance & Validation
- Evaluated on test set using Accuracy, Macro-F1, Precision, and Recall.
- Plotted training loss and validation convergence across epochs.
- Validated on 6 qualitative inference sentences covering diverse automotive sentiments.

---

## 5. Execution Instructions
```bash
cd Assignments/Assignment_08
jupyter notebook TTL_Assignment_08.ipynb
```
