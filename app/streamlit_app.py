"""Human or AI? Text Origin Miner: Streamlit app (DS-311 Data Mining, Group 7).
Page 1 (M3): paste text -> human / AI probability and the words that drove the prediction."""
import html, re, unicodedata
from pathlib import Path

import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Text Origin Miner", page_icon="🔍")

# Same cleaning as the training data (copied from the team notebooks)
NAV_LINES = {
    "The New York Times", "Site Navigation", "Site Search Navigation", "Site Mobile Navigation",
    "Slideshow controls", "Recent Episodes", "What's Next", "transcript", "Covid-19Guidance", "Archive",
    "Supported by", "United States", "Credits", "Data", "Nursing Homes", "Vaccines", "Treatments",
    "Global Vaccinations", "Deaths Above Normal", "Colleges and Universities", "Your County’s Risk",
    "Hospitals Near You", "Your Places",
}
# Template lines in AI texts such as "By [Your Name]", "Published: [Date]", "By Your Name"
PLACEHOLDER_LINE = re.compile(
    r"^\s*(?:by|written by|published|date|author)\b[^\n]{0,30}\[[^\]\n]{1,40}\][^\n]{0,20}$|^\s*by your name\s*$", re.I)

def clean_text(text):
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = re.sub(r"<[^<>\n]{1,50}>", " ", text)                    # HTML tags
    text = re.sub(r"\*\*|__", "", text)                              # markdown bold
    text = re.sub(r"(?m)^[ \t]*#{1,6}[ \t]*", "", text)              # markdown headings
    lines = [l for l in text.split("\n")
             if l.strip() not in NAV_LINES and not PLACEHOLDER_LINE.match(l)]
    text = re.sub(r"^\s*(?:Comments\b|Title:)\s*", "", "\n".join(lines))   # leading "Comments" / "Title:"
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


@st.cache_resource
def load_detector():
    return joblib.load(Path(__file__).parent / "models" / "detector.joblib")


detector = load_detector()
vectorizer, classifier = detector[0], detector[-1]
terms = np.array(vectorizer.get_feature_names_out())
weights = classifier.coef_[0]                 # positive -> AI, negative -> human
ai_index = list(classifier.classes_).index(1)

st.title("Human or AI? Text Origin Miner")
st.caption("DS-311 Data Mining · Group 7 · research demo")
st.warning("AI-text detectors make mistakes, including on genuine human writing. "
           "This is a course demo: never use it to accuse anyone of using AI.")

text = st.text_area("Paste a news-style text (at least 50 words):", height=260)

if st.button("Analyse", type="primary"):
    cleaned = clean_text(text)
    n_words = len(cleaned.split())
    if n_words < 50:
        st.error(f"Please paste at least 50 words (after cleaning: {n_words}).")
    else:
        p_ai = float(detector.predict_proba([cleaned])[0][ai_index])
        verdict = "Likely AI-generated" if p_ai >= 0.5 else "Likely human-written"
        st.subheader(verdict)
        st.metric("Probability of AI", f"{p_ai:.0%}")
        st.progress(p_ai)

        contribution = vectorizer.transform([cleaned]).toarray()[0] * weights
        present = np.nonzero(contribution)[0]
        to_ai = [terms[i] for i in present[np.argsort(contribution[present])[::-1]] if contribution[i] > 0][:12]
        to_human = [terms[i] for i in present[np.argsort(contribution[present])] if contribution[i] < 0][:12]
        left, right = st.columns(2)
        left.markdown("**Words pointing to AI**")
        left.write(", ".join(to_ai) or "none")
        right.markdown("**Words pointing to human**")
        right.write(", ".join(to_human) or "none")
        st.caption(f"{n_words} words analysed · TF-IDF + logistic regression trained on the Defactify dataset")
