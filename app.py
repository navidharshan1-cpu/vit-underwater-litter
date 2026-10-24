"""Police contact narrative classifier: a fine-tuned DeBERTa-v3-base that reads a Boston Police FIO narrative
and predicts the circumstance of the contact (Stopped / Encountered / Observed).

Run locally with:  streamlit run app.py
"""
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------- settings
GITHUB_REPO = "navidharshan1-cpu/police-narrative-classifier"   # GitHub "username/repository"
RELEASE_TAG = "v1.0"                                             # Release holding model.safetensors
WEIGHTS = "model.safetensors"
MODEL_FILES = ["config.json", "tokenizer.json", "tokenizer_config.json"]   # small files kept in the repository

ROOT = Path(__file__).parent
LABELS = ["Stopped", "Encountered", "Observed"]
ICONS = {"Stopped": "🛑", "Encountered": "💬", "Observed": "👁️"}
MEANING = {
    "Stopped": "The officer **stopped or detained** the person, e.g. a traffic stop or a subject stop.",
    "Encountered": "The officer **interacted or spoke with** the person, without a formal stop.",
    "Observed": "The officer **only observed** the person, without stopping or speaking to them.",
}

# Results from the notebook (used when metrics.json is not in the repository)
DEFAULT = {
    "accuracy": 0.815, "macro_f1": 0.748, "weighted_f1": 0.819, "macro_auc": 0.922,
    "auc": {"Stopped": 0.937, "Encountered": 0.884, "Observed": 0.945},
    "best_epoch": 4, "best_val_macro_f1": 0.763, "train_minutes": 8.8,
}
DATA = {"rows": 4860, "narratives": 4618, "counts": {"Stopped": 2917, "Encountered": 1075, "Observed": 626},
        "split": {"Train": 3232, "Validation": 693, "Test": 693}, "median_tokens": 152, "truncated": "1.0%"}

st.set_page_config(page_title="Police Narrative Classifier", page_icon="🚔", layout="wide")


def find(name):
    for p in (ROOT / name, ROOT / "model" / name, ROOT / "results" / name, ROOT / "figures" / name):
        if p.exists():
            return p
    return None


def show_fig(name, caption):
    p = find(name)
    if p:
        st.image(str(p), caption=caption, use_container_width=True)
    return p is not None


def clean(text):
    """Same cleaning as in training: redaction marks -> [REDACTED], officer names -> [NAME]."""
    t = str(text)
    t = re.sub(r'X{2,}(?:/X{2,})*', ' [REDACTED] ', t)
    title = r"(?:Officers?|Ofc\.?|P\.?O\.?|Sgt\.?|Sergeant|Det\.?|Detective|Lt\.?|Lieutenant|Trooper|Capt\.?|Captain)"
    name = r"(?!(?:and|of|the|in|on|at)\b)[A-Z][a-zA-Z'\-]+(?:\s+[A-Z][a-zA-Z'\-]+)?"
    for _ in range(2):
        t = re.sub(rf"\b({title})\s+(?!\[NAME\]){name}(\s+(?:and|&)\s+(?!{title}\b){name})?", r'\1 [NAME]', t)
    t = re.sub(r'(\[REDACTED\]\s*){2,}', '[REDACTED] ', t)
    return re.sub(r'\s+', ' ', t).strip()


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def load_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = ROOT / "model_files"
    model_dir.mkdir(exist_ok=True)
    for f in MODEL_FILES:
        src = find(f)
        if src is None:
            raise FileNotFoundError(f"{f} is missing from the repository.")
        (model_dir / f).write_bytes(src.read_bytes())
    weights = find(WEIGHTS)
    if weights is None:
        weights = model_dir / WEIGHTS
        if not weights.exists():
            tmp = weights.with_suffix(".part")
            urllib.request.urlretrieve(
                f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{WEIGHTS}", tmp)
            tmp.rename(weights)
    elif weights.parent != model_dir:
        (model_dir / WEIGHTS).write_bytes(weights.read_bytes())

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).float().eval()
    torch.set_num_threads(2)
    return tok, model


def get_model():
    with st.spinner("Loading the model (first time only, about a minute)..."):
        return load_model()


def predict(tok, model, texts):
    import torch
    with torch.no_grad():
        enc = tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt")
        return model(**enc).logits.float().softmax(-1).numpy()


def word_importance(tok, model, text, k, max_words=120):
    words = text.split()[:max_words]
    base = predict(tok, model, [" ".join(words)])[0, k]
    drops = []
    variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
    for i in range(0, len(variants), 16):
        drops.extend(base - predict(tok, model, variants[i:i + 16])[:, k])
    return words, np.array(drops)


def highlight(words, drops):
    top = max(drops.max(), 1e-6)
    out = []
    for w, d in zip(words, drops):
        a = max(0.0, d) / top
        out.append(f'<span style="background: rgba(232,131,58,{a:.2f}); padding: 1px 2px; border-radius: 3px">{w}</span>'
                   if a > 0.15 else w)
    return " ".join(out)


# ---------------------------------------------------------------- results
@st.cache_data
def load_results():
    f = find("test_results.csv")
    if f is None:
        return None
    r = pd.read_csv(f, index_col=0)
    r.index = r.index.astype(str).str.strip()
    return r


@st.cache_data
def load_metrics():
    m = dict(DEFAULT)
    f = find("metrics.json")
    if f:
        try:
            j = json.loads(f.read_text())
            for k in ("accuracy", "macro_f1", "weighted_f1", "macro_auc"):
                if isinstance(j.get(k), (int, float)):
                    m[k] = float(j[k])
        except Exception:
            pass
    return m


def headline(r):
    def num(row, col):
        try:
            return float(r.loc[row, col])
        except (KeyError, ValueError, TypeError):
            return None
    return {"accuracy": num("accuracy", "F1"), "macro_f1": num("macro average", "F1"),
            "per_class": {l: {c: num(l, c) for c in ("precision", "recall", "F1")} for l in LABELS if l in r.index},
            "support": {l: num(l, r.columns[-1]) for l in LABELS if l in r.index}}


@st.cache_data
def load_examples():
    f = find("examples.csv")
    return pd.read_csv(f) if f else None


results = load_results()
H = headline(results) if results is not None else None
M = load_metrics()
ACC = H["accuracy"] if H and H["accuracy"] is not None else M["accuracy"]
MF1 = H["macro_f1"] if H and H["macro_f1"] is not None else M["macro_f1"]


def show_prediction(probs, truth=None):
    k = int(probs.argmax())
    left, right = st.columns([1, 1.3])
    with left:
        st.subheader(f"{ICONS[LABELS[k]]} {LABELS[k]}")
        st.metric("Confidence", f"{probs[k]:.1%}")
        st.caption(MEANING[LABELS[k]].replace("**", ""))
    with right:
        for i, l in enumerate(LABELS):
            st.progress(float(probs[i]), text=f"{ICONS[l]} {l}: {probs[i]:.1%}")
    with st.container(border=True):
        if truth is not None:
            st.markdown(f"**Accuracy on this narrative:** true circumstance **{truth}**, predicted **{LABELS[k]}**: "
                        + ("✅ correct" if truth == LABELS[k] else "❌ incorrect"))
        else:
            st.markdown("**Accuracy on this narrative:** not measurable (no true label for new text)")
        pc = H["per_class"].get(LABELS[k], {}) if H else {}
        st.markdown(f"**Model accuracy on 693 unseen test narratives:** accuracy **{ACC:.1%}** · macro F1 **{MF1:.1%}**"
                    + (f"  \nFor **{LABELS[k]}** predictions: precision {pc.get('precision', 0):.1%} · "
                       f"recall {pc.get('recall', 0):.1%} · F1 {pc.get('F1', 0):.1%}" if pc else ""))
        if probs[k] < 0.6:
            st.warning("Low confidence: the narrative fits more than one circumstance. Treat this prediction with care.")
    return k


# ---------------------------------------------------------------- sidebar navigation
PAGES = ["Overview Dashboard", "Live Narrative Analysis", "Example Narrative Gallery", "Model Performance",
         "Project & Workflow", "Dataset & Preprocessing", "Use Cases & Impact", "About & Limitations"]
with st.sidebar:
    st.markdown("## 🚔 Police Narrative Classifier")
    st.caption("Fine-tuned DeBERTa-v3-base · NLP")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    st.metric("Test accuracy", f"{ACC:.1%}")
    st.metric("Macro F1", f"{MF1:.1%}")
    st.caption("Classifies a police contact narrative as **Stopped**, **Encountered** or **Observed**.")


# ================================================================ pages
if page == "Overview Dashboard":
    st.title("🚔 Police Contact Narrative Classifier")
    st.markdown(
        "A fine-tuned **DeBERTa-v3-base** transformer reads the free-text narrative an officer wrote about a "
        "Boston Police **Field Interrogation and Observation (FIO)** contact and classifies the **circumstance** of the contact.")
    c1, c2, c3 = st.columns(3)
    for col, l in zip((c1, c2, c3), LABELS):
        col.info(f"**{ICONS[l]} {l}**  \n{MEANING[l]}")

    st.subheader("Key results (held-out test set)")
    m = st.columns(4)
    m[0].metric("Accuracy", f"{ACC:.1%}")
    m[1].metric("Macro F1", f"{MF1:.1%}")
    m[2].metric("Weighted F1", f"{M['weighted_f1']:.1%}")
    m[3].metric("Macro ROC-AUC", f"{M['macro_auc']:.3f}")
    m = st.columns(4)
    m[0].metric("Narratives", f"{DATA['narratives']:,}")
    m[1].metric("Test narratives", "693")
    m[2].metric("Parameters", "184 M")
    m[3].metric("Training time", f"{M['train_minutes']} min (T4 GPU)")

    st.subheader("F1 score per circumstance")
    if H:
        st.bar_chart(pd.DataFrame({"F1": {l: v["F1"] for l, v in H["per_class"].items()}}), horizontal=True)
    st.markdown("➡️ Use **Live Narrative Analysis** in the sidebar to classify your own narrative.")

elif page == "Live Narrative Analysis":
    st.title("🔍 Live Narrative Analysis")
    st.markdown("Write or paste an FIO-style narrative. The model predicts the circumstance and its confidence.")
    text = st.text_area("Narrative", height=170, placeholder=(
        "e.g. Officers conducted a traffic stop on a black Honda Accord for an expired inspection sticker. "
        "The operator was calm and cordial. Verbal warning issued."))
    explain = st.checkbox("Show which words influenced the prediction (slower, ~10-30 s)")
    if st.button("Classify", type="primary", disabled=not text.strip()):
        tok, model = get_model()
        cleaned = clean(text)
        probs = predict(tok, model, [cleaned])[0]
        k = show_prediction(probs)
        if cleaned != text.strip():
            with st.expander("Text after cleaning (as the model sees it)"):
                st.write(cleaned)
        if explain:
            with st.spinner("Measuring word importance..."):
                words, drops = word_importance(tok, model, cleaned, k)
            st.markdown("**Words that pushed the model towards this prediction** (darker = more influence):")
            st.markdown(highlight(words, drops), unsafe_allow_html=True)
            if len(cleaned.split()) > len(words):
                st.caption(f"Only the first {len(words)} words are analysed.")

elif page == "Example Narrative Gallery":
    st.title("🗂️ Example Narrative Gallery")
    st.markdown("Real narratives from the **test set** (never seen in training). The true circumstance is known, "
                "so you can check whether the model is right.")
    examples = load_examples()
    if examples is None or not len(examples):
        st.warning("examples.csv not found in the repository.")
    else:
        if st.button("▶️ Classify all examples", type="primary"):
            tok, model = get_model()
            probs = predict(tok, model, [clean(t) for t in examples.text.astype(str)])
            st.session_state["gallery"] = probs
        probs = st.session_state.get("gallery")
        if probs is not None and len(probs) == len(examples):
            preds = [LABELS[int(p.argmax())] for p in probs]
            correct = sum(p == t for p, t in zip(preds, examples.circumstance))
            st.success(f"Accuracy on these {len(examples)} examples: **{correct}/{len(examples)} "
                       f"({correct / len(examples):.0%})**")
        for l in LABELS:
            sub = examples[examples.circumstance == l]
            if not len(sub):
                continue
            st.subheader(f"{ICONS[l]} True circumstance: {l}")
            for i, r in sub.iterrows():
                with st.container(border=True):
                    st.write(str(r.text))
                    if probs is not None and len(probs) == len(examples):
                        p = probs[i]
                        k = int(p.argmax())
                        ok = LABELS[k] == l
                        st.markdown(f"**Predicted:** {ICONS[LABELS[k]]} {LABELS[k]} ({p[k]:.1%}) "
                                    + ("✅ correct" if ok else "❌ incorrect"))

elif page == "Model Performance":
    st.title("📊 Model Performance")
    st.caption("Held-out test set: 693 narratives (15% of the data) never used for training or model selection.")
    m = st.columns(4)
    m[0].metric("Accuracy", f"{ACC:.1%}")
    m[1].metric("Macro F1", f"{MF1:.1%}")
    m[2].metric("Weighted F1", f"{M['weighted_f1']:.1%}")
    m[3].metric("Macro ROC-AUC", f"{M['macro_auc']:.3f}")
    if H:
        rows = [{"Circumstance": f"{ICONS[l]} {l}", "Precision": v["precision"], "Recall": v["recall"], "F1": v["F1"],
                 "ROC-AUC": M["auc"].get(l), "Test narratives": int(H["support"][l] or 0)}
                for l, v in H["per_class"].items()]
        st.dataframe(pd.DataFrame(rows).set_index("Circumstance").style.format(
            {"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}", "ROC-AUC": "{:.3f}"}), use_container_width=True)
    else:
        st.warning("test_results.csv not found in the repository.")
    st.markdown("- **Stopped** is recognised most reliably: its narratives use distinctive language "
                "(\"traffic stop\", \"T/S\", \"operator\").\n"
                "- **Encountered** and **Observed** are harder: many narratives of every type say \"officers observed…\", "
                "so these circumstances overlap in wording. Most confident mistakes are Encountered narratives "
                "predicted as Observed or Stopped.\n"
                f"- Best checkpoint: epoch {M['best_epoch']} (validation macro F1 {M['best_val_macro_f1']:.3f}).")
    figs = [("fig3_confusion_matrix.png", "Confusion matrix"), ("fig3b_per_class_scores.png", "Per-class scores"),
            ("fig3c_roc_pr_curves.png", "ROC and precision-recall curves"), ("fig3d_confidence.png", "Confidence of predictions"),
            ("fig2_training_curves.png", "Training curves"), ("fig4_bias_analysis.png", "Accuracy by subgroup"),
            ("fig5_word_importance.png", "Most influential words")]
    shown = [f for f in figs if find(f[0])]
    for i in range(0, len(shown), 2):
        cols = st.columns(2)
        for col, (f, cap) in zip(cols, shown[i:i + 2]):
            with col:
                show_fig(f, cap)

elif page == "Project & Workflow":
    st.title("🧭 Project & Workflow")
    st.markdown("""
**Goal:** fine-tune one well-chosen transformer to classify the circumstance of a police contact from the officer's narrative.

**Why DeBERTa-v3-base?** It improves on BERT/RoBERTa with *disentangled attention* (content and position are encoded
separately) and an enhanced mask decoder, and it is among the strongest base-size encoders for text classification.
It reads up to 512 tokens, which covers 99% of the narratives, and fine-tunes in under 10 minutes on a free T4 GPU.
""")
    steps = [
        ("1. Load data", "Boston Police FIO 2024 CSV (4,860 rows)."),
        ("2. Clean", "Remove duplicate rows of multi-person contacts, mark redactions as [REDACTED], mask officer names as [NAME]."),
        ("3. Split", "Stratified 70 / 15 / 15: train 3,232 · validation 693 · test 693."),
        ("4. Tokenise", "DeBERTa-v3 tokenizer, max 512 tokens, dynamic padding."),
        ("5. Fine-tune", "AdamW lr 2e-5, 10% warm-up + linear decay, batch 8 × 2 accumulation, 4 epochs, mixed precision, class-weighted cross-entropy."),
        ("6. Select", "Keep the epoch with the best validation macro F1."),
        ("7. Evaluate", "Test accuracy, precision/recall/F1, confusion matrix, ROC/PR curves, confidence, subgroup analysis."),
        ("8. Explain", "Occlusion word importance: remove each word and measure the drop in probability."),
        ("9. Deploy", "Weights on GitHub Releases, this Streamlit app on Streamlit Community Cloud."),
    ]
    for t, d in steps:
        with st.container(border=True):
            st.markdown(f"**{t}** · {d}")
    show_fig("fig2_training_curves.png", "Training and validation curves")

elif page == "Dataset & Preprocessing":
    st.title("📁 Dataset & Preprocessing")
    st.markdown("**Source:** Boston Police Department *Field Interrogation and Observation (FIO)* records, 2024 "
                "(Analyze Boston open data). Each record contains the officer's free-text narrative and the recorded circumstance.")
    m = st.columns(4)
    m[0].metric("Rows in CSV", f"{DATA['rows']:,}")
    m[1].metric("Unique narratives", f"{DATA['narratives']:,}")
    m[2].metric("Median length", f"{DATA['median_tokens']} tokens")
    m[3].metric("Truncated at 512", DATA["truncated"])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Class distribution**")
        counts = pd.DataFrame({"Narratives": DATA["counts"]})
        st.bar_chart(counts)
        st.caption("Stopped is 63% of the data, so the loss is class-weighted and macro F1 is the main metric.")
    with c2:
        st.markdown("**Data split (stratified)**")
        st.dataframe(pd.DataFrame({"Narratives": DATA["split"]}), use_container_width=True)
    st.markdown("""
**Preprocessing**
- **De-duplication:** a contact with several people appears as several rows with the same narrative; keeping one per
  contact number (`fc_num`) prevents the same text appearing in both training and test.
- **Redactions:** runs of `XXX` in the source are replaced by `[REDACTED]`.
- **Officer names:** names after titles (Officer, Sgt., Det., …) are replaced by `[NAME]`, so the model learns
  wording rather than who wrote the report.
- **No data augmentation** was used: the narratives are real records, and paraphrasing could change the circumstance.
""")
    show_fig("fig1_data.png", "Class distribution and narrative lengths")
    show_fig("fig1b_token_lengths.png", "Token lengths")

elif page == "Use Cases & Impact":
    st.title("🌍 Use Cases & Impact")
    st.markdown("""
**Where a model like this can help (research and oversight, not decisions about people):**
- **Data-quality checks:** flag records whose narrative does not match the recorded circumstance, so analysts can review them.
- **Transparency and accountability research:** label large archives of narratives consistently to study how stops,
  encounters and observations are recorded over time.
- **Analyst support:** sort and search thousands of free-text reports faster than reading them by hand.
- **Teaching:** a worked example of fine-tuning a transformer on real, sensitive public-sector text.

**Impact**
- Automates a reading task that takes a person minutes per report, in under a second.
- The confidence score and word highlighting show *why* a prediction was made, supporting human review.
""")
    st.warning("This model must **not** be used to make or justify decisions about individuals, policing actions or legal outcomes.")

elif page == "About & Limitations":
    st.title("ℹ️ About & Limitations")
    st.markdown(f"""
**Model:** `microsoft/deberta-v3-base` (184 M parameters), fine-tuned end-to-end on 3,232 narratives
(AdamW lr 2e-5, class-weighted cross-entropy, max 512 tokens). Best checkpoint by validation macro F1.

**Test results:** accuracy {ACC:.1%} · macro F1 {MF1:.1%} · macro ROC-AUC {M['macro_auc']:.3f}.

**Limitations**
- The label is the circumstance **recorded by the officer**; the model learns how officers write, not what happened.
- **Encountered** and **Observed** overlap in wording, so they are confused more often than Stopped.
- Accuracy differs between subgroups (e.g. 68% for contacts with basis *Encounter* vs 90% for *Probable Cause*).
- Trained on one city and one year; other departments or later years may use different language.
- The model can be confidently wrong. It is a research and education tool.

**Responsible use:** narratives contain sensitive information. Names were redacted in the source data and officer
names are masked before classification. Text typed into this app is not stored.
""")
    st.caption("NLP project · University of Wollongong in Dubai")
