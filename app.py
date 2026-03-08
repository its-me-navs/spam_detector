import streamlit as st
import pickle
import numpy as np

# ── Page config ──────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🚨",
    layout="centered"
)

# ── Dark theme styling ────────────────────────────────────
st.markdown("""
    <style>
        body { background-color: #0e1117; }
        .result-spam {
            background-color: #3d0000;
            border-left: 5px solid #ff4b4b;
            padding: 15px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            color: #ff4b4b;
        }
        .result-ham {
            background-color: #003d1a;
            border-left: 5px solid #00c853;
            padding: 15px;
            border-radius: 8px;
            font-size: 20px;
            font-weight: bold;
            color: #00c853;
        }
        .model-card {
            background-color: #1e2130;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────
@st.cache_resource
def load_models():
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    with open('model1_nb.pkl', 'rb') as f:
        model1 = pickle.load(f)
    with open('model2_lr.pkl', 'rb') as f:
        model2 = pickle.load(f)
    with open('model3_svm.pkl', 'rb') as f:
        model3 = pickle.load(f)
    return tfidf, model1, model2, model3

tfidf, model1, model2, model3 = load_models()

models = {
    "Naive Bayes": model1,
    "Logistic Regression": model2,
    "Linear SVC": model3
}

# ── UI ────────────────────────────────────────────────────
st.title("🚨 Spam Message Detector")
st.markdown("Type a message below and see what each model thinks.")

message = st.text_area("Enter a message", height=120, 
                        placeholder="e.g. Win a FREE iPhone now!")

if st.button("Analyse", use_container_width=True):
    if message.strip() == "":
        st.warning("Please enter a message first.")
    else:
        msg_v = tfidf.transform([message])
        
        st.markdown("---")
        st.subheader("Results")

        votes = {"Ham": 0, "Spam": 0}

        for name, model in models.items():
            pred = model.predict(msg_v)[0]
            label = "Spam" if pred == 1 else "Ham"
            votes[label] += 1

            with st.container():
                st.markdown(f"**{name}**")

                col1, col2 = st.columns([1, 2])

                with col1:
                    if label == "Spam":
                        st.markdown('<div class="result-spam">🚨 Spam</div>', 
                                    unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="result-ham">✅ Ham</div>', 
                                    unsafe_allow_html=True)

                with col2:
                    if hasattr(model, "predict_proba"):
                        prob = model.predict_proba(msg_v)[0]
                        st.markdown("**Probability**")
                        st.progress(float(prob[1]), 
                                    text=f"Spam: {prob[1]:.0%}  |  Ham: {prob[0]:.0%}")
                    else:
                        st.markdown("*(Linear SVC does not support probabilities)*")

                st.markdown("---")


        st.subheader("🗳️ Verdict")
        if votes["Spam"] >= 2:
            st.error(f"**SPAM** — {votes['Spam']}/3 models agree")
        else:
            st.success(f"**HAM** — {votes['Ham']}/3 models agree")