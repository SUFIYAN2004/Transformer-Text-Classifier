#  Transformer-Based News Topic Classifier

A high-performance text classification system built using a custom **Transformer architecture** (Multi-Head Attention) from scratch. This project classifies news headlines into four categories: Business, Science & Technology, Sports, and World News with **92% accuracy**.

##  Features
*   **Custom Transformer Architecture:** Built using the Keras Functional API without high-level boilerplate.
*   **Learned Positional Encoding:** Injects sequence order information via additive position embeddings.
*   **Optimized Training:** Implements Cosine Learning Rate Decay, Label Smoothing, and Class Weight balancing.
*   **Live Streamlit Demo:** A "Hacker-style" web interface for real-time headline analysis.

##  Model Performance
The model was evaluated using a comprehensive classification report:

| Category | F1-Score | Precision | Recall |
| :--- | :--- | :--- | :--- |
| **World News** | 0.95 | 0.96 | 0.94 |
| **Sports** | 0.95 | 0.93 | 0.97 |
| **Science & Tech** | 0.88 | 0.89 | 0.87 |
| **Business** | 0.85 | 0.82 | 0.88 |

**Overall Accuracy:** 92%

## Tech Stack
*   **Framework:** TensorFlow 2.x / Keras
*   **Architecture:** Multi-Head Attention, Layer Normalization, Global Average Pooling
*   **Frontend:** Streamlit
*   **Data Processing:** NLTK (Lemmatization), Scikit-Learn (Label Encoding)

##  Installation & Usage
1. **Clone the repository:**
   ```bash
   git clone https://github.com/SUFIYAN2004/Transformer-Text-Classifier.git
   cd Transformer-Text-Classifier
   ```

2. **Install dependencies:**
   ```bash
   pip install tensorflow streamlit pandas numpy scikit-learn nltk
   ```

3. **Run the Streamlit App:**
   ```bash
   streamlit run app.py
   ```

##  Architecture Overview
The model follows the Transformer encoder logic:
1. **Input Layer:** Tokenized sequences (Max length: 37).
2. **Embedding + Positional Addition:** Merging word semantics with sequence position.
3. **Multi-Head Attention:** 2 heads with a key dimension of 64.
4. **Global Average Pooling:** Reducing 3D attention maps to a classification vector.
5. **Dense Layers:** ReLU hidden layer (64 units) with Dropout (0.3) for regularization.
6. **Softmax Output:** 4-class probability distribution.

##  Author
**V. Mohammed Sufiyan**  
*Computer Applications Student*

