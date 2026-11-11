"""
Fake News Detector - screen
Run:  pip install pandas scikit-learn joblib streamlit
      pip install python-docx pypdf          (optional - to upload Word and PDF files)
      streamlit run app.py
Keep fake_news_detector.py (and your data files) in the same folder as this file.
"""
import io
import os

import pandas as pd
import streamlit as st

import fake_news_detector as fnd

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

# The verdict "stamp" is the one styled element; everything else is plain Streamlit.
st.markdown("""
<style>
.stamp {display:inline-block; margin:8px 0 14px; padding:10px 28px; border:5px double currentColor;
        border-radius:6px; font-family:Georgia, 'Times New Roman', serif; font-size:44px;
        font-weight:700; letter-spacing:4px; transform:rotate(-4deg);}
.stamp.fake {color:#c62828;}
.stamp.real {color:#2e7d32;}
.word {display:inline-block; margin:3px 4px; padding:2px 10px; border-radius:12px; font-size:15px;}
.word.fake {background:rgba(198,40,40,.12); color:#c62828;}
.word.real {background:rgba(46,125,50,.12); color:#2e7d32;}
@media (prefers-reduced-motion: no-preference) {.stamp {animation:press .25s ease-out;}}
@keyframes press {from {transform:rotate(-4deg) scale(1.4); opacity:0;} to {transform:rotate(-4deg) scale(1); opacity:1;}}
</style>
""", unsafe_allow_html=True)

EXAMPLES = {
    "Example: real-style article": "The city council approved a new budget on Tuesday that increases "
                                   "funding for public schools and road repairs over the next two years.",
    "Example: fake-style article": "SHOCKING! Doctors don't want you to know this one fruit cures every "
                                   "disease overnight. Share before they delete it!",
}


def data_source():
    if os.path.exists(os.path.join(fnd.HERE, "Fake.csv")) and os.path.exists(os.path.join(fnd.HERE, "True.csv")):
        return "Fake.csv + True.csv"
    if os.path.exists(os.path.join(fnd.HERE, "news.csv")):
        return "news.csv"
    return "built-in demo data"


@st.cache_resource(show_spinner="Training the model...")
def get_model(source):
    """Train once and keep the model in memory. 'source' is part of the cache key, so adding
    a data file and clicking Retrain trains on the new data."""
    model, results = fnd.build_and_evaluate(fnd.load_data())
    fnd.joblib.dump(model, fnd.MODEL_FILE)
    return model, results


def explain(model, article, n=6):
    """Words in this article that pushed the prediction towards FAKE or towards REAL."""
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["classifier"]
    x = tfidf.transform([fnd.clean_text(article)]).toarray()[0]
    contrib = x * clf.coef_[0]                     # > 0 points to classes_[1], < 0 to classes_[0]
    words = tfidf.get_feature_names_out()
    present = [(words[i], contrib[i]) for i in x.nonzero()[0]]
    to_high = [w for w, c in sorted(present, key=lambda p: -p[1]) if c > 0][:n]
    to_low = [w for w, c in sorted(present, key=lambda p: p[1]) if c < 0][:n]
    return (to_low, to_high) if clf.classes_[0] == "FAKE" else (to_high, to_low)


def read_uploaded_text(uploaded):
    """Text of an uploaded .txt, .docx or .pdf file. Returns (text, error message)."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".txt"):
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                return data.decode(enc), None
            except UnicodeDecodeError:
                continue
    if name.endswith(".docx"):
        try:
            from docx import Document
        except ImportError:
            return "", "To read Word files, install python-docx:  python -m pip install python-docx"
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs), None
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            return "", "To read PDF files, install pypdf:  python -m pip install pypdf"
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            return "", "No text found in this PDF - it is probably a scanned image."
        return text, None
    return "", "This file type is not supported."


def show_result(article):
    """The verdict stamp, confidence and the words behind the decision."""
    if not article.strip():
        st.warning("There is no text to check.")
        return
    if not fnd.clean_text(article):
        st.warning("The text has no words the model can read. Try a longer piece of text.")
        return
    label, confidence = fnd.predict(model, article)
    kind = "fake" if label == "FAKE" else "real"
    st.markdown('<div class="stamp {}">{}</div>'.format(kind, label), unsafe_allow_html=True)
    st.write("The model is **{:.0%}** sure this article is **{}**.".format(confidence, label.lower()))
    st.progress(float(confidence))

    fake_words, real_words = explain(model, article)
    w1, w2 = st.columns(2)
    with w1:
        st.markdown("**Words that point to fake**")
        st.markdown(" ".join('<span class="word fake">{}</span>'.format(w) for w in fake_words)
                    or "None found", unsafe_allow_html=True)
    with w2:
        st.markdown("**Words that point to real**")
        st.markdown(" ".join('<span class="word real">{}</span>'.format(w) for w in real_words)
                    or "None found", unsafe_allow_html=True)
    st.caption("A prediction is a hint, not proof. Always check the source of a story.")


def text_column(df):
    """The column that holds the articles in an uploaded CSV."""
    for name in ("text", "article", "content", "body", "news", "title", "headline"):
        for col in df.columns:
            if str(col).strip().lower() == name:
                return col
    text_cols = [c for c in df.columns if df[c].dtype == object]
    return max(text_cols, key=lambda c: df[c].astype(str).str.len().mean()) if text_cols else None


# ----------------------------------------------------------------------------------------------
source = data_source()
with st.sidebar:
    st.header("Model")
    st.write("Training data: **{}**".format(source))
    if source == "built-in demo data":
        st.info("The demo data is tiny, so predictions are rough. Put the Kaggle files "
                "Fake.csv and True.csv in this folder and click Retrain.")
    if st.button("Retrain model"):
        get_model.clear()
model, results = get_model(source)
with st.sidebar:
    st.metric("Accuracy on test articles", "{:.1%}".format(results["accuracy"]))
    st.caption("Tested on {} articles the model never saw while learning.".format(results["test_size"]))

st.title("📰 Fake News Detector")
st.write("Paste a news article or a headline to see whether it looks fake or real.")

tab_check, tab_model = st.tabs(["Check an article", "How the model performs"])

with tab_check:
    how = st.radio("How do you want to give the article?", ["Paste text", "Upload a file"], horizontal=True)

    if how == "Paste text":
        c1, c2, _ = st.columns([1, 1, 3])
        for col, (label, text) in zip((c1, c2), EXAMPLES.items()):
            with col:
                if st.button(label, use_container_width=True):
                    st.session_state.article = text
        article = st.text_area("Article or headline", key="article", height=200,
                               placeholder="Paste the article text here...")
        if st.button("Check article", type="primary"):
            show_result(article)

    else:
        uploaded = st.file_uploader(
            "Upload an article (.txt, .docx, .pdf) or a CSV with many articles",
            type=["txt", "docx", "pdf", "csv"])
        if uploaded is not None and uploaded.name.lower().endswith(".csv"):
            # Many articles at once: one prediction per row, downloadable as CSV
            try:
                df = pd.read_csv(uploaded)
            except Exception:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, encoding="latin-1")
            col = text_column(df)
            if col is None:
                st.error("No text column found. Name the column with the articles 'text'.")
            else:
                st.write("Checking **{}** articles from the column **{}**.".format(len(df), col))
                preds = [fnd.predict(model, t) if fnd.clean_text(t) else ("-", 0.0)
                         for t in df[col].fillna("").astype(str)]
                out = df.copy()
                out["prediction"] = [p[0] for p in preds]
                out["confidence"] = ["{:.0%}".format(p[1]) if p[0] != "-" else "" for p in preds]
                counts = out["prediction"].value_counts()
                k1, k2 = st.columns(2)
                k1.metric("Predicted fake", int(counts.get("FAKE", 0)))
                k2.metric("Predicted real", int(counts.get("REAL", 0)))
                st.dataframe(out[[col, "prediction", "confidence"]], use_container_width=True, height=400)
                st.download_button("Download results (CSV)", out.to_csv(index=False).encode("utf-8"),
                                   file_name="predictions_" + uploaded.name, mime="text/csv")
        elif uploaded is not None:
            text, error = read_uploaded_text(uploaded)
            if error:
                st.error(error)
            else:
                with st.expander("Text read from the file ({} words)".format(len(text.split()))):
                    st.write(text[:3000] + ("..." if len(text) > 3000 else ""))
                if st.button("Check article", type="primary"):
                    show_result(text)

with tab_model:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Articles", results["articles"])
    m2.metric("Used to learn", results["train_size"])
    m3.metric("Used to test", results["test_size"])
    m4.metric("Accuracy", "{:.1%}".format(results["accuracy"]))

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        st.caption("How many test articles were classified correctly (diagonal) and wrongly.")
        st.dataframe(results["confusion"], use_container_width=True)
        st.subheader("Articles per label")
        st.bar_chart(results["counts"])
    with right:
        st.subheader("Precision and recall")
        st.dataframe(results["report_table"].round(2), use_container_width=True)
        st.subheader("Words the model relies on most")
        st.markdown("**Fake:** " + ", ".join(results["fake_words"]))
        st.markdown("**Real:** " + ", ".join(results["real_words"]))

    st.subheader("How it works")
    st.markdown(
        "1. **Clean** the text: lowercase, remove links, numbers and symbols.\n"
        "2. **TF-IDF** turns each article into numbers: how important each word is in it.\n"
        "3. **Logistic Regression** learns which words point to fake and which to real news, "
        "using 80% of the articles.\n"
        "4. The other 20% are used to **test** it, which gives the accuracy shown above.")
