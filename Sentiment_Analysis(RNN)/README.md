# 🎬 IMDB Movie Review Sentiment Analysis using LSTM

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Keras](https://img.shields.io/badge/Keras-DeepLearning-red?style=for-the-badge&logo=keras)
![NLP](https://img.shields.io/badge/NLP-Sentiment%20Analysis-green?style=for-the-badge)
![LSTM](https://img.shields.io/badge/Model-LSTM-purple?style=for-the-badge)

### Predict whether a movie review is Positive 😊 or Negative 😞 using Deep Learning

</div>

---

# 📌 Project Overview

This project performs **Sentiment Analysis** on IMDB movie reviews using a **Long Short-Term Memory (LSTM)** neural network.

The model learns patterns from movie reviews and predicts whether a review expresses a:

- 😊 Positive Sentiment
- 😞 Negative Sentiment

Natural Language Processing (NLP) techniques such as tokenization, sequence padding, and word embeddings are used to convert textual data into numerical form before training the LSTM model.

---

# 🎯 Objectives

- Perform sentiment analysis on movie reviews
- Learn text preprocessing techniques
- Implement tokenization and padding
- Build and train an LSTM model
- Evaluate model performance
- Predict sentiment for custom user reviews

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Python | Programming Language |
| Pandas | Data Processing |
| NumPy | Numerical Operations |
| TensorFlow | Deep Learning Framework |
| Keras | Neural Network API |
| Scikit-Learn | Data Splitting & Evaluation |
| Matplotlib | Visualization |

---

# 📂 Dataset

Dataset Used:

**IMDB Movie Reviews Dataset**

The dataset contains movie reviews labeled as:

| Sentiment | Meaning |
|------------|------------|
| Positive | Good Review |
| Negative | Bad Review |

Dataset Structure:

| Review | Sentiment |
|----------|----------|
| Amazing movie with great acting | Positive |
| Worst film I have ever watched | Negative |

---

# ⚙️ Project Workflow

```text
Raw Reviews
      │
      ▼
Text Cleaning
      │
      ▼
Tokenization
      │
      ▼
Padding
      │
      ▼
Train-Test Split
      │
      ▼
Embedding Layer
      │
      ▼
LSTM Layer
      │
      ▼
Dense Layers
      │
      ▼
Sentiment Prediction
```

---

# 📖 Data Preprocessing

## 1. Tokenization

Text reviews are converted into numerical sequences.

Example:

Review:

```text
I love this movie
```

After Tokenization:

```text
[15, 78, 22, 101]
```

---

## 2. Vocabulary Limitation

```python
MAX_WORDS = 10000
```

Only the 10,000 most frequent words are kept.

---

## 3. Sequence Padding

```python
MAX_LEN = 200
```

All reviews are converted to a fixed length of 200 words.

Example:

```text
[15, 78, 22]
```

Becomes:

```text
[15, 78, 22, 0, 0, 0, ...]
```

---

# 🧠 Model Architecture

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=32),
    LSTM(64),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

## Architecture Diagram

```text
Input Review
      │
      ▼
Embedding Layer
(10000 → 32 Dimensions)
      │
      ▼
LSTM Layer
(64 Units)
      │
      ▼
Dropout Layer
(50%)
      │
      ▼
Dense Layer
(16 Neurons + ReLU)
      │
      ▼
Output Layer
(1 Neuron + Sigmoid)
      │
      ▼
Positive / Negative
```

---

# 🔍 Layer Explanation

## Embedding Layer

Converts words into dense vectors.

Example:

```text
movie → [0.12, 0.44, 0.78, ...]
```

Helps the model understand relationships between words.

---

## LSTM Layer

Long Short-Term Memory (LSTM) is a special type of Recurrent Neural Network (RNN).

Advantages:

- Remembers context
- Handles long sentences
- Captures word order
- Better than traditional RNNs

---

## Dropout Layer

```python
Dropout(0.5)
```

Randomly disables 50% neurons during training.

Purpose:

- Prevent Overfitting
- Improve Generalization

---

## Dense Layer

Learns higher-level sentiment patterns extracted by the LSTM.

---

## Sigmoid Output Layer

Produces a probability between:

```text
0 → Negative
1 → Positive
```

Decision Rule:

```python
if prediction >= 0.5:
    Positive
else:
    Negative
```

---

# 📈 Model Training

Model Compilation:

```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

Training:

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_split=0.2
)
```

---

# 💾 Saving Best Model

```python
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True
)
```

The best performing model is automatically saved during training.

---

# 📊 Model Evaluation

Metrics Used:

- Accuracy
- Loss
- Validation Accuracy
- Validation Loss

Example:

```text
Test Accuracy: 75.00%
```

---

# 🔮 Making Predictions

Example Review:

```text
This movie was fantastic and full of emotions.
```

Prediction:

```text
Positive 😊
Confidence: 0.92
```

Example Review:

```text
Worst movie ever. Complete waste of time.
```

Prediction:

```text
Negative 😞
Confidence: 0.04
```

---

# 📁 Project Structure

```text
IMDB-Sentiment-Analysis/
│
├── IMDB Dataset.csv
├── sentiment_analysis.ipynb
├── best_model.keras
├── README.md
│
├── models/
│   └── best_model.keras
│
├── screenshots/
│   ├── training.png
│   └── prediction.png
│
└── requirements.txt
```

---

# 🚀 Future Improvements

- Bidirectional LSTM
- GRU Networks
- Attention Mechanism
- Transformer Models
- BERT Sentiment Analysis
- Streamlit Web Application
- Hyperparameter Tuning

---

# 📚 Learning Outcomes

Through this project, the following concepts were learned:

- Natural Language Processing
- Text Tokenization
- Sequence Padding
- Word Embeddings
- Recurrent Neural Networks
- LSTM Architecture
- Binary Classification
- Model Evaluation
- Deep Learning with TensorFlow

---

# 🤝 Author

### Anam Mulla

Deep Learning & Machine Learning Enthusiast

---

# ⭐ If you found this project useful, consider giving it a star!

```
⭐ Star this repository
🍴 Fork the project
🧠 Learn Deep Learning
🚀 Keep Building
```

---
