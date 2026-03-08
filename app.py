import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from model import load_models, load_data, predict
from model import get_confusion_matrix, get_classification_report, get_top_words

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Spam Detector",
    page_icon="🚨",
    layout="centered"
)

# ── Styling ───────────────────────────────────────────────
st.markdown("""
    <style>
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
    </style>
""", unsafe_allow_html=True)

# ── Load everything ───────────────────────────────────────
@st.cache_resource
def setup():
    tfidf, model1, model2, model3 = load_models()
    x_train, x_test, y_train, y_test = load_data()
    x_train_v = tfidf.transform(x_train)
    x_test_v = tfidf.transform(x_test)
    return tfidf, model1, model2, model3, x_test_v, y_test

tfidf, model1, model2, model3, x_test_v, y_test = setup()

models = {
    "Naive Bayes": model1,
    "Logistic Regression": model2,
    "Linear SVC": model3
}

# ── Tabs ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Predict", "📊 Analysis"])

# ════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ════════════════════════════════════════════════════════
with tab1:
    st.title("🚨 Spam Message Detector")
    st.markdown("Type a message and see what each model thinks.")

    message = st.text_area("Enter a message", height=120,
                            placeholder="e.g. Claim your free goodies!")

    if st.button("Analyse", use_container_width=True):
        if message.strip() == "":
            st.warning("Please enter a message first.")
        else:
            st.markdown("---")
            st.subheader("Results")

            votes = {"Ham": 0, "Spam": 0}

            for name, model in models.items():
                label, prob = predict(model, tfidf, message)
                votes[label] += 1

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
                    if prob is not None:
                        st.markdown("**Probability**")
                        st.progress(float(prob[1]),
                                    text=f"Spam: {prob[1]:.0%}  |  Ham: {prob[0]:.0%}")
                    else:
                        st.markdown("*(Linear SVC does not support probabilities)*")

                st.markdown("---")

            # Verdict
            st.subheader("🗳️ Verdict")
            if votes["Spam"] >= 2:
                st.error(f"**SPAM** — {votes['Spam']}/3 models agree")
            else:
                st.success(f"**HAM** — {votes['Ham']}/3 models agree")

# ════════════════════════════════════════════════════════
# TAB 2 — ANALYSIS
# ════════════════════════════════════════════════════════
with tab2:
    st.title("📊 Model Analysis")

    # ── Confusion Matrices ────────────────────────────────
    st.subheader("Confusion Matrices")
    cols = st.columns(3)

    for col, (name, model) in zip(cols, models.items()):
        cm = get_confusion_matrix(model, x_test_v, y_test)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'],
                    cmap='Reds', ax=ax)
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.tight_layout()
        col.pyplot(fig)
        plt.close()

    # ── Classification Reports ────────────────────────────
    st.markdown("---")
    st.subheader("Classification Reports")

    for name, model in models.items():
        df = get_classification_report(model, x_test_v, y_test)
        st.markdown(f"**{name}**")
        st.dataframe(df, use_container_width=True)
        st.markdown("")

    # ── Top Words ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("Top Spam vs Ham Words (Naive Bayes)")

    words = get_top_words(tfidf, model1)
    col1, col2 = st.columns(2)

    # Spam words
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(words["feature_names"][words["top_spam_idx"]][::-1],
            words["diff_weights"][words["top_spam_idx"]][::-1],
            color='#ff4b4b')
    ax.set_title("Top 20 Spam Words")
    ax.set_xlabel("Score")
    plt.tight_layout()
    col1.pyplot(fig)
    plt.close()

    # Ham words
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(words["feature_names"][words["top_ham_idx"]][::-1],
            words["diff_weights"][words["top_ham_idx"]][::-1],
            color='#00c853')
    ax.set_title("Top 20 Ham Words")
    ax.set_xlabel("Score")
    plt.tight_layout()
    col2.pyplot(fig)
    plt.close()