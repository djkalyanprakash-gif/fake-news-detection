"""
Fake News Detection — Streamlit app.
Student project (BTech CSE, Data Science) by Kalyan.

Run:
    python train_model.py   # once, after putting the datasets in data/
    streamlit run app.py

If model/fake_news_model.joblib is missing, the app tries the base64
copy at model/fake_news_model.joblib.b64. If neither is present it
still works: it trains a small in-memory fallback on built-in samples
and says so clearly on screen, instead of crashing.

The model only knows 2016–2018 training data, so it cannot judge
recent events. The optional live check searches Google News for the
headline and shows real recent coverage next to the model verdict.
If a Gemini API key is set in Streamlit secrets (GEMINI_API_KEY),
Gemini briefly summarises whether that coverage supports the
headline. Without a key the live check still works — it just shows
the sources without a summary. The live check is off when the user
unticks it, and then nothing leaves the app.
"""

import base64
import html
import io
import json
import re
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "model" / "fake_news_model.joblib"
MODEL_B64_PATH = BASE / "model" / "fake_news_model.joblib.b64"
METRICS_PATH = BASE / "model" / "metrics.json"

EXAMPLES = {
    "Real — metro announcement": (
        "Government announces new metro line for Hyderabad",
        "The state government on Tuesday announced a new metro rail line connecting "
        "the airport to the city centre. Officials said construction will begin next "
        "year and the project is expected to take three years to complete."),
    "Fake — miracle cure": (
        "Miracle cure found: drink this and never get sick again",
        "Doctors don't want you to know this secret drink cures every disease in "
        "24 hours. Big pharma is hiding it. One glass a day and you will never "
        "need a hospital again!"),
    "Fake — shocking alien claim": (
        "SHOCKING! Scientists prove aliens built the pyramids overnight",
        "SECRET documents leaked by an insider PROVE that aliens built the pyramids "
        "in one night and the government has been hiding it for 5000 years. "
        "Share before they delete this!"),
}

FALLBACK_SAMPLES = [
    ("Government announces new metro line for Hyderabad The state government announced a new metro rail line.", 0),
    ("Scientists confirm water found on the Moon Researchers said new readings confirm water ice in craters.", 0),
    ("RBI keeps repo rate unchanged at 6.5 percent The Reserve Bank kept the repo rate unchanged on Friday.", 0),
    ("SHOCKING Scientists prove aliens built the pyramids overnight SECRET documents leaked by an insider PROVE it.", 1),
    ("Miracle cure found drink this and never get sick again Doctors don't want you to know this secret drink.", 1),
    ("You won't BELIEVE what this actor did doctors hate him This one weird trick will make you rich overnight.", 1),
]


def clean(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_model():
    """Return (model, metrics, is_fallback). Never raises on missing model.

    Tries, in order: the binary model file, the base64 copy of it
    (kept in the repo because binary files cannot be pushed here),
    then a small in-memory sample fallback.
    """
    metrics = {}
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH), metrics, False
    if MODEL_B64_PATH.exists():
        raw = base64.b64decode(MODEL_B64_PATH.read_text())
        return joblib.load(io.BytesIO(raw)), metrics, False
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
    ])
    texts = [clean(t) for t, _ in FALLBACK_SAMPLES]
    labels = [label for _, label in FALLBACK_SAMPLES]
    model.fit(texts, labels)
    return model, {"source": "sample_fallback", "accuracy": None}, True


def top_words(model, text: str, n: int = 8):
    """Words that pushed the prediction most, from the linear model."""
    try:
        vec = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        x = vec.transform([clean(text)])
        coefs = clf.coef_[0]
        scored = sorted(
            ((coefs[i], vec.get_feature_names_out()[i])
             for i in x.indices if abs(coefs[i]) > 0),
            key=lambda t: -abs(t[0]))
        return [(w, round(float(c), 3)) for c, w in scored[:n]]
    except Exception:
        return []


def live_news_search(query: str, max_results: int = 5):
    """Search Google News (RSS, no key needed) for recent coverage.

    Returns (results, error): results is a list of dicts with
    title, link, source and published; error is a short message
    when the search itself failed, else None.
    """
    query = " ".join(str(query or "").split())[:200]
    if not query:
        return [], "No headline to search for."
    url = ("https://news.google.com/rss/search?q="
           f"{quote_plus(query)}&hl=en-IN&gl=IN&ceid=IN:en")
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; NewsCheck/1.0)"},
            timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
    except Exception:
        return [], ("Live news search is unavailable right now — "
                    "check your connection and try again.")
    results = []
    for item in root.iter("item"):
        title = html.unescape((item.findtext("title") or "").strip())
        link = (item.findtext("link") or "").strip()
        if not title or not link:
            continue
        # Google News titles end with " - SourceName"; split it off.
        source = ""
        if " - " in title:
            title, source = title.rsplit(" - ", 1)
        source_el = item.find("source")
        if source_el is not None and source_el.text:
            source = source_el.text.strip()
        published = ""
        pub = item.findtext("pubDate")
        if pub:
            try:
                published = parsedate_to_datetime(pub).strftime("%d %b %Y")
            except Exception:
                published = ""
        results.append({"title": title, "link": link,
                        "source": source, "published": published})
        if len(results) >= max_results:
            break
    return results, None


def get_gemini_key():
    """Gemini API key from Streamlit secrets, or None if not set."""
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        return key.strip() if isinstance(key, str) and key.strip() else None
    except Exception:
        return None


def gemini_assessment(api_key: str, headline: str, results):
    """Ask Gemini whether the found coverage supports the headline.

    Returns a short plain-language assessment, or None on any failure.
    The model is told to judge only from the listed coverage, not
    from its own knowledge.
    """
    coverage = "\n".join(
        f"- {r['title']} ({r['source'] or 'unknown source'}"
        f"{', ' + r['published'] if r['published'] else ''})"
        for r in results)
    prompt = (
        "Headline to check: " + headline.strip() + "\n\n"
        "Recent news coverage found for this headline:\n" + coverage + "\n\n"
        "In two or three short sentences, say whether this coverage supports "
        "the headline, contradicts it, or is unclear — judging ONLY from the "
        "articles listed above, not from your own knowledge. If the coverage "
        "does not actually cover the headline's claim, say so plainly."
    )
    try:
        resp = requests.post(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            "gemini-2.5-flash:generateContent",
            params={"key": api_key},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=20)
        resp.raise_for_status()
        data = resp.json()
        text = (data["candidates"][0]["content"]["parts"][0]["text"]
                .strip())
        return text or None
    except Exception:
        return None


def show_live_check(headline: str, article: str):
    """Render the live news check section for the given headline."""
    query = headline.strip() or " ".join(article.split()[:12])
    st.subheader("Live news check")
    st.caption("Searched current coverage for this headline. This sends "
               "only the headline to Google News — untick the box above "
               "to keep everything on-device.")
    with st.spinner("Checking live news coverage…"):
        results, error = live_news_search(query)
    if error:
        st.warning(error)
        return
    if not results:
        st.info("No recent coverage found for this headline. That does not "
                "prove it is false — it may be too new, too local, or "
                "worded differently. Try the exact headline from the "
                "article you saw.")
        return
    for r in results:
        meta = " · ".join(p for p in (r["source"], r["published"]) if p)
        st.markdown(f"— [{r['title']}]({r['link']})"
                    + (f"  \n  <span class='muted'>{meta}</span>"
                       if meta else ""),
                    unsafe_allow_html=True)
    st.caption("Read the coverage above to judge whether the claim is "
               "actually true — the model verdict is only about writing "
               "style, not facts.")
    api_key = get_gemini_key()
    if api_key:
        with st.spinner("Gemini is reading the coverage…"):
            verdict = gemini_assessment(api_key, query, results)
        if verdict:
            st.write("**Gemini's read of the coverage:**")
            st.write(verdict)


st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.markdown("""
<style>
  .block-container { max-width: 760px; padding-top: 2.2rem; }
  h1 { font-family: Georgia, serif; font-size: 2rem; margin-bottom: 0; }
  .kicker { letter-spacing: .14em; text-transform: uppercase; font-size: .72rem; color: #8a6d3b; }
  .verdict-real { border-left: 5px solid #1a7f37; background: #f0f7f1; padding: 1rem 1.2rem; }
  .verdict-fake { border-left: 5px solid #b42318; background: #fdf1f0; padding: 1rem 1.2rem; }
  .muted { color: #5c5c5c; font-size: .9rem; }
</style>
""", unsafe_allow_html=True)

model, metrics, is_fallback = load_model()

st.markdown('<div class="kicker">Student project · Machine learning</div>', unsafe_allow_html=True)
st.title("Fake News Detection")
st.write("Paste a news headline and article. The model estimates whether it "
         "reads like real or fake news, and shows how confident it is. "
         "It works best on a full article — a one-line question or a headline "
         "on its own gives an unreliable result, and the model cannot check "
         "facts or recent events.")

if is_fallback:
    st.warning("Demo mode — the trained model file was not found, so this is "
               "running on a small built-in sample. Run `python train_model.py` "
               "with the Kaggle datasets in `data/` for the full model.")
elif metrics:
    acc = metrics.get("accuracy")
    rows = metrics.get("rows")
    st.caption(f"Model: {metrics.get('model', '')}"
               + (f" · Held-out accuracy: {acc:.1%}" if acc else "")
               + (f" · Trained on {rows:,} articles" if rows else ""))

if "history" not in st.session_state:
    st.session_state.history = []
if "headline" not in st.session_state:
    st.session_state.headline = ""
if "article" not in st.session_state:
    st.session_state.article = ""


def _fill_example():
    choice = st.session_state.example_choice
    if choice in EXAMPLES:
        st.session_state.headline, st.session_state.article = EXAMPLES[choice]


st.selectbox("Try an example", ["— Write your own —", *EXAMPLES],
             key="example_choice", on_change=_fill_example)

with st.form("analyse"):
    st.text_input("Headline", key="headline")
    st.text_area("Article text", height=180, key="article",
                 placeholder="Paste the full article here…")
    live_check = st.checkbox(
        "Also check live news coverage for this headline",
        value=True,
        help="Searches Google News for the headline and shows real recent "
             "coverage next to the model verdict. Sends only the headline "
             "to Google News. Untick to keep everything on-device.")
    submitted = st.form_submit_button("Analyse article", type="primary")

headline = st.session_state.headline
article = st.session_state.article

if submitted:
    full = f"{headline} {article}".strip()
    if len(clean(full)) < 20:
        if headline.strip() and live_check:
            st.info("Too little text for the model — it needs a headline "
                    "and at least a few sentences. Showing the live news "
                    "check for the headline instead.")
            show_live_check(headline, article)
        else:
            st.error("Please paste a headline and at least a few sentences of article text.")
    else:
        proba = model.predict_proba([clean(full)])[0]
        fake_prob = float(proba[1])
        verdict = "Fake" if fake_prob >= 0.5 else "Real"
        confidence = fake_prob if verdict == "Fake" else 1 - fake_prob
        cls = "verdict-fake" if verdict == "Fake" else "verdict-real"
        word_count = len(clean(full).split())
        if word_count < 40:
            st.warning(
                f"Short text — only {word_count} words. The model judges "
                "writing style, and with this little text the verdict is "
                "little better than a guess. Paste the full article for a "
                "meaningful result.")
        st.markdown(
            f'<div class="{cls}"><b>Verdict: Likely {verdict} news</b><br>'
            f'Confidence: {confidence:.0%} &nbsp;·&nbsp; '
            f'Fake probability: {fake_prob:.0%}</div>',
            unsafe_allow_html=True)
        st.progress(confidence, text=f"{confidence:.0%} confident it is {verdict.lower()}")
        words = top_words(model, full)
        if words:
            st.write("**Words that influenced this result most:**")
            st.write(", ".join(f"`{w}` ({c:+.2f})" for w, c in words))
        st.caption("A high score means the writing resembles fake articles in the "
                   "training data — it is not proof that a claim is false. "
                   "The model was trained on 2016–2018 articles and knows "
                   "nothing about recent events, so it cannot tell you whether "
                   "something that happened this year is true. "
                   "Always check the original source.")
        st.session_state.history.insert(0, {
            "headline": headline or "(no headline)",
            "verdict": verdict, "confidence": round(confidence, 2)})
        st.session_state.history = st.session_state.history[:5]
        if live_check:
            st.divider()
            show_live_check(headline, article)

if st.session_state.history:
    st.subheader("Recent checks")
    for item in st.session_state.history:
        st.write(f"— {item['headline']} · **{item['verdict']}** ({item['confidence']:.0%})")
    if st.button("Clear history"):
        st.session_state.history = []
        st.rerun()

st.divider()
st.markdown('<p class="muted">Trained on two Kaggle datasets combined: Fake/True news '
            '(23,481 fake + 21,417 real articles) and a 2,260-article fake/real text '
            'collection. Mostly 2016–2018 US and celebrity news, so results are '
            'weakest on recent or non-English stories. Built by Kalyan, BTech CSE '
            '(Data Science), 3rd year — a student project, not a fact-checker.</p>',
            unsafe_allow_html=True)
