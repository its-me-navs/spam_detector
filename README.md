# SMS Spam Detector

A machine learning web app that classifies SMS messages as **spam** or **ham** using three different models — with a comparison dashboard to analyze their performance.

**Live Demo**: [spam-detector.streamlit.app](https://spam-detector-navs.streamlit.app/) 

---

##  What It Does

- Takes any SMS message as input and predicts whether it's spam or ham
- Runs the message through **3 trained models** simultaneously
- Shows a **majority vote verdict** with confidence scores
- Includes an **Analysis tab** with confusion matrices, classification reports, and top spam/ham word visualizations

---

## Model Performance

Trained on the [UCI SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) (5,169 messages after deduplication).

| Model | Accuracy | Spam Recall |
|---|---|---|
| Multinomial Naive Bayes | 97% | 0.74 |
| Logistic Regression | 96% | 0.68 |
| **Linear SVC** | **98%** | **0.89** |

Linear SVC performs best — especially for spam recall, which matters most in this task (missing spam is worse than a false alarm).

---

## Tech Stack

- **Python** — scikit-learn, pandas, numpy
- **NLP** — TF-IDF vectorization
- **Models** — MultinomialNB, LogisticRegression, LinearSVC
- **App** — Streamlit
- **Visualization** — Matplotlib, Seaborn

---

## Key Learnings

- Handling **class imbalance** (7:1 ham:spam ratio) — using `class_weight='balanced'` on SVC boosted spam recall from 0.74 → 0.89
- Why **recall matters more than accuracy** for spam detection
- How **TF-IDF** converts text into meaningful numerical features
- Building a clean multi-model comparison UI with Streamlit