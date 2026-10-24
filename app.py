"""ViT Underwater Litter: classify underwater objects (trash / bio / rov) with a fine-tuned Vision Transformer.

Streamlit app for the MAAI901 Applied Vision Transformer report. Run locally with:  streamlit run app.py
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------- settings
GITHUB_REPO = "navidharshan1-cpu/vit-underwater-litter"   # GitHub "username/repository"
RELEASE_TAG = "v1.0"                                       # Release that holds the model file
WEIGHTS = "vit_b16_trashcan_fp16.pt"
BASE_MODEL = "google/vit-base-patch16-224"

ROOT = Path(__file__).parent
NAMES = ["trash", "bio", "rov"]
LABELS = {"trash": "🗑️ Trash (litter)", "bio": "🐟 Bio (marine life)", "rov": "🤖 ROV (robot parts)"}
MEANING = {"trash": "Human-made litter on the sea floor: plastic, bags, bottles, nets, cans, wood and more.",
           "bio": "Marine life: fish, crabs, starfish, shells, eels and plants.",
           "rov": "Parts of the robot (ROV) itself, such as its arm or frame, seen by its own camera."}

# Results from the notebook (test set: 1,975 objects from 56 unseen videos)
RESULTS = pd.DataFrame({
    "accuracy": [0.856, 0.808, 0.589, 0.442], "macro F1": [0.855, 0.808, 0.585, 0.204],
    "trash F1": [0.841, 0.783, 0.479, 0.613], "bio F1": [0.832, 0.774, 0.593, 0.000], "rov F1": [0.894, 0.868, 0.683, 0.000],
}, index=["ViT fine-tuned", "ViT frozen + linear", "CLIP zero-shot (no training)", "Majority class (reference)"])
PER_CLASS = pd.DataFrame({"Precision": [0.847, 0.920, 0.830], "Recall": [0.834, 0.759, 0.969],
                          "F1": [0.841, 0.832, 0.894], "Test objects": [873, 498, 604]}, index=NAMES)
COUNTS = pd.DataFrame({"train": [4159, 1940, 2288], "val": [954, 353, 405], "test": [873, 498, 604]}, index=NAMES)

st.set_page_config(page_title="ViT Underwater Litter", page_icon="🔬", layout="wide")

# ---------------------------------------------------------------- style
st.markdown("""
<style>
.hero {background: linear-gradient(120deg, #03254c 0%, #0a4f7a 55%, #1287a8 100%); color: #fff;
       padding: 28px 32px; border-radius: 16px; margin-bottom: 18px;}
.hero h1 {color: #fff; margin: 0 0 6px 0; font-size: 2.2rem;}
.hero p {color: #d7eef7; margin: 0; font-size: 1.05rem;}
.card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 16px 18px; height: 100%;
       background: rgba(18,135,168,.06);}
.card h4 {margin: 0 0 6px 0;}
.kpi {border-radius: 14px; padding: 14px 16px; background: rgba(18,135,168,.10); border-left: 5px solid #1287a8;}
.kpi .v {font-size: 1.7rem; font-weight: 700;}
.kpi .l {font-size: .85rem; opacity: .75;}
.step {border-left: 4px solid #1287a8; padding: 8px 14px; margin-bottom: 8px; border-radius: 6px;
       background: rgba(18,135,168,.06);}
</style>""", unsafe_allow_html=True)


def hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


def kpis(items):
    for col, (v, l) in zip(st.columns(len(items)), items):
        col.markdown(f'<div class="kpi"><div class="v">{v}</div><div class="l">{l}</div></div>', unsafe_allow_html=True)
    st.write("")


def find_file(name):
    for p in (ROOT / name, ROOT / "results" / name, ROOT / "models" / name, ROOT / "figures" / name):
        if p.exists():
            return p
    return None


def show_fig(name, caption):
    p = find_file(name)
    if p:
        st.image(str(p), caption=caption, use_container_width=True)
    return p is not None


# ---------------------------------------------------------------- model
@st.cache_resource(show_spinner=False)
def load_model():
    import torch
    from transformers import ViTConfig, ViTForImageClassification

    path = find_file(WEIGHTS)
    if path is None:
        path = ROOT / WEIGHTS
        url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{WEIGHTS}"
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(path)

    config = ViTConfig.from_pretrained(BASE_MODEL, num_labels=len(NAMES), id2label=dict(enumerate(NAMES)),
                                       label2id={n: i for i, n in enumerate(NAMES)})
    model = ViTForImageClassification(config)
    try:
        model.set_attn_implementation("eager")      # needed to read the attention maps
    except Exception:
        try:
            model.config._attn_implementation = "eager"
        except Exception:
            pass                                   # app still works; the attention map is then skipped
    state = torch.load(path, map_location="cpu")
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in state.items()})
    model.eval()
    torch.set_num_threads(2)
    return model


def get_model():
    with st.spinner("Loading the ViT (first time only, about a minute)..."):
        return load_model()


def preprocess(img):
    import torch
    x = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).permute(2, 0, 1)[None]


def classify(model, img):
    """Returns class probabilities and the attention-rollout map (14 x 14, 0..1)."""
    import torch
    with torch.no_grad():
        out = model(pixel_values=preprocess(img), output_attentions=True)
    probs = out.logits.softmax(-1)[0].numpy()
    heat = None
    if out.attentions and out.attentions[0] is not None:
        att = torch.stack(out.attentions).squeeze(1).mean(1)      # layers x tokens x tokens, heads averaged
        att = att + torch.eye(att.size(-1))
        att = att / att.sum(-1, keepdim=True)
        R = att[0]
        for a in att[1:]:
            R = a @ R
        heat = R[0, 1:].reshape(14, 14).numpy()
        heat = heat / heat.max()
    return probs, heat


def overlay(img, heat, alpha=0.45):
    import matplotlib.cm as cm
    base = img.convert("RGB").resize((448, 448))
    h = Image.fromarray(np.uint8(heat * 255)).resize((448, 448), Image.BICUBIC)
    colored = Image.fromarray(np.uint8(cm.jet(np.asarray(h) / 255.0)[..., :3] * 255))
    return Image.blend(base, colored, alpha)


@st.cache_data
def load_results():
    f = find_file("test_results.csv")
    if f is None:
        return RESULTS
    try:
        r = pd.read_csv(f, index_col=0)
        return r if "ViT fine-tuned" in r.index else RESULTS
    except Exception:
        return RESULTS


def samples():
    files = sorted(set(ROOT.glob("sample__*.jpg")) | set((ROOT / "samples").glob("sample__*.jpg")))
    return {f"{f.name.split('__')[1]} · {f.name.split('__', 2)[2]}": f for f in files}


results = load_results()
VIT = results.loc["ViT fine-tuned"]


def show_result(img, probs, heat, truth=None):
    k = int(probs.argmax())
    left, mid, right = st.columns([1, 1, 1.2])
    left.image(img, caption="Input object", use_container_width=True)
    if heat is not None:
        mid.image(overlay(img, heat), caption="Attention rollout: where the ViT looks (red = most attention)",
                  use_container_width=True)
    with right:
        st.subheader(f"Prediction: {LABELS[NAMES[k]]}")
        st.metric("Confidence", f"{probs[k]:.1%}")
        for i, c in enumerate(NAMES):
            st.progress(float(probs[i]), text=f"{LABELS[c]}: {probs[i]:.1%}")
        st.caption(MEANING[NAMES[k]])
    with st.container(border=True):
        if truth is not None:
            st.markdown(f"**Accuracy on this image:** true class **{truth}**, predicted **{NAMES[k]}**: "
                        + ("✅ correct" if truth == NAMES[k] else "❌ incorrect"))
        else:
            st.markdown("**Accuracy on this image:** not measurable (no true label for an uploaded photo)")
        pc = PER_CLASS.loc[NAMES[k]]
        st.markdown(f"**Model accuracy on 1,975 unseen test objects:** accuracy **{VIT['accuracy']:.1%}** · "
                    f"macro F1 **{VIT['macro F1']:.1%}**  \nFor **{NAMES[k]}** predictions: precision "
                    f"{pc['Precision']:.1%} · recall {pc['Recall']:.1%} · F1 {pc['F1']:.1%}")
        if probs[k] < 0.6:
            st.warning("Low confidence: the object may be unclear, too small, or not one of the three classes.")


# ---------------------------------------------------------------- sidebar navigation
PAGES = ["Overview Dashboard", "Live Image Analysis", "Sample Image Gallery", "Model Performance",
         "Explainability (Attention)", "Project & Workflow", "Dataset & Augmentation", "Use Cases & Impact",
         "About & Limitations"]
with st.sidebar:
    st.markdown("## 🔬 ViT Underwater Litter")
    st.caption("Vision Transformer · ViT-B/16 · TrashCan 1.0")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    st.metric("Test accuracy", f"{VIT['accuracy']:.1%}")
    st.metric("Macro F1", f"{VIT['macro F1']:.1%}")
    st.caption("Classifies one underwater object as **trash**, **bio** or **rov**.")


# ================================================================ pages
if page == "Overview Dashboard":
    hero("🔬 ViT Underwater Litter",
         "A fine-tuned Vision Transformer that recognises marine litter in deep-sea robot (ROV) footage.")
    st.markdown("The model looks at **one object** in an underwater image and decides what it is:")
    for col, c in zip(st.columns(3), NAMES):
        col.markdown(f'<div class="card"><h4>{LABELS[c]}</h4>{MEANING[c]}</div>', unsafe_allow_html=True)
    st.write("")
    st.subheader("Key results on the test set")
    kpis([(f"{VIT['accuracy']:.1%}", "Test accuracy"), (f"{VIT['macro F1']:.1%}", "Macro F1"),
          ("+4.7 pts", "macro F1 vs frozen ViT"), ("+27 pts", "macro F1 vs CLIP zero-shot")])
    kpis([("12,074", "Object crops"), ("1,975", "Test objects (56 unseen videos)"),
          ("85.8 M", "Parameters"), ("~9 min", "Fine-tuning on a T4 GPU")])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**F1 score per class (fine-tuned ViT)**")
        st.bar_chart(PER_CLASS[["F1"]], horizontal=True)
    with c2:
        st.markdown("**Macro F1 by method**")
        st.bar_chart(results[["macro F1"]], horizontal=True)
    st.info("➡️ Open **Live Image Analysis** in the sidebar to classify your own underwater image.")

elif page == "Live Image Analysis":
    hero("🔍 Live Image Analysis", "Upload an underwater image. The ViT predicts the class and shows where it looked.")
    up = st.file_uploader("Underwater image", type=["jpg", "jpeg", "png", "webp"])
    if up is None:
        st.info("Upload an image of **one underwater object** (or use the crop sliders to frame one object). "
                "No image? Try the **Sample Image Gallery**.")
    else:
        img = Image.open(up).convert("RGB")
        W, H = img.size
        with st.expander("✂️ Crop to one object (optional)"):
            c1, c2 = st.columns(2)
            x0, x1 = c1.slider("Left / right (%)", 0, 100, (0, 100))
            y0, y1 = c2.slider("Top / bottom (%)", 0, 100, (0, 100))
        if x1 - x0 >= 5 and y1 - y0 >= 5:
            img = img.crop((int(W * x0 / 100), int(H * y0 / 100), int(W * x1 / 100), int(H * y1 / 100)))
        model = get_model()
        probs, heat = classify(model, img)
        show_result(img, probs, heat)

elif page == "Sample Image Gallery":
    hero("🖼️ Sample Image Gallery", "Real objects from the test set (unseen dives). The true class is known.")
    ex = samples()
    if not ex:
        st.warning("No example images (sample__*.jpg) found in the repository.")
    else:
        if st.button("▶️ Classify all samples", type="primary"):
            model = get_model()
            st.session_state["gallery"] = {k: classify(model, Image.open(f).convert("RGB"))[0] for k, f in ex.items()}
        preds = st.session_state.get("gallery")
        if preds:
            ok = sum(NAMES[int(p.argmax())] == k.split(" · ")[0] for k, p in preds.items())
            st.success(f"Accuracy on these {len(preds)} samples: **{ok}/{len(preds)} ({ok / len(preds):.0%})**")
        for c in NAMES:
            items = [(k, f) for k, f in ex.items() if k.startswith(c + " ·")]
            if not items:
                continue
            st.subheader(f"True class: {LABELS[c]}")
            cols = st.columns(4)
            for i, (k, f) in enumerate(items):
                with cols[i % 4]:
                    st.image(str(f), use_container_width=True)
                    if preds and k in preds:
                        p = preds[k]
                        j = int(p.argmax())
                        st.markdown(f"**{NAMES[j]}** ({p[j]:.0%}) " + ("✅" if NAMES[j] == c else "❌"))
        st.divider()
        st.subheader("Inspect one sample in detail")
        key = st.selectbox("Sample", list(ex))
        if st.button("Analyse this sample"):
            img = Image.open(ex[key]).convert("RGB")
            probs, heat = classify(get_model(), img)
            show_result(img, probs, heat, truth=key.split(" · ")[0])

elif page == "Model Performance":
    hero("📊 Model Performance", "Test set: 1,975 objects from 56 deep-sea videos never used in training or model selection.")
    kpis([(f"{VIT['accuracy']:.1%}", "Accuracy"), (f"{VIT['macro F1']:.1%}", "Macro F1"),
          ("86.6%", "Macro precision"), ("85.4%", "Macro recall")])
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.markdown("**Per-class results (fine-tuned ViT)**")
        t = PER_CLASS.copy()
        t.index = [LABELS[c] for c in t.index]
        st.dataframe(t.style.format({"Precision": "{:.1%}", "Recall": "{:.1%}", "F1": "{:.1%}"}),
                     use_container_width=True)
    with c2:
        st.markdown("**Comparison of methods**")
        cols = [c for c in ["accuracy", "macro F1", "trash F1", "bio F1", "rov F1"] if c in results.columns]
        st.dataframe(results[cols].style.format("{:.3f}").highlight_max(axis=0, color="#1287a855"),
                     use_container_width=True)
    st.markdown("""
- **Fine-tuning helps:** training all ViT layers raises macro F1 from 0.808 (frozen features) to **0.855**.
- **CLIP zero-shot** (text prompts only, no training) reaches 0.585, so domain-specific training matters under water.
- **ROV** parts are recognised best (recall 96.9%); **bio** has high precision (92%) but some marine life is missed.
- **Robustness:** accuracy 83% on small objects (<32 px) vs 88% on large ones; per-video accuracy ranges from 41% to 100%.
""")
    for f, cap in [("fig3_confusion_matrices.png", "Confusion matrices"), ("fig2_training_curves.png", "Training curves"),
                   ("fig4_bias_analysis.png", "Accuracy by object size, clarity and video")]:
        show_fig(f, cap)

elif page == "Explainability (Attention)":
    hero("🧠 Explainability", "Attention rollout shows which image patches the ViT relied on for its decision.")
    st.markdown("""
A ViT splits the image into **196 patches** (14 × 14) of 16 × 16 pixels. In each of its 12 layers, every patch attends
to every other patch. **Attention rollout** (Abnar & Zuidema, 2020) multiplies the attention of all layers to estimate
how much each patch contributed to the final [CLS] decision. Red areas in the map are where the model looked most.

- When the model is right, attention usually sits **on the object** itself.
- Mistakes can happen when attention spreads over the **background** (sand, murky water) or the object is tiny.
""")
    shown = show_fig("fig5_attention_maps.png", "Attention rollout on test objects")
    shown |= show_fig("fig6_mistakes.png", "Most confident mistakes")
    if not shown:
        st.info("Try it live: open **Live Image Analysis** or the **Sample Image Gallery**; every prediction shows its attention map.")

elif page == "Project & Workflow":
    hero("🧭 Project & Workflow", "From raw ROV video frames to a deployed Vision Transformer.")
    st.markdown("**Why ViT-B/16?** Self-attention relates every image patch to every other patch, so the model can use the "
                "whole scene (sea floor, lighting, the robot) as context. ImageNet pre-training gives strong general "
                "features, and fine-tuning adapts them to murky deep-sea images.")
    steps = [
        ("1. Data", "TrashCan 1.0: 7,212 frames from JAMSTEC deep-sea ROV videos with object annotations."),
        ("2. Object crops", "Each annotated object cut out with 10% context; 22 classes merged into trash / bio / rov."),
        ("3. Video-level split", "Whole videos kept in one split (train 203, val 51, test 56) so test dives are unseen."),
        ("4. Pre-processing", "Resize to 224 × 224, normalise; augmentation for training."),
        ("5. Baseline", "Frozen ViT [CLS] embeddings + logistic regression (macro F1 0.808)."),
        ("6. Fine-tuning", "All 86 M weights, AdamW lr 3e-5, one-cycle schedule, batch 32, 5 epochs, class-weighted loss."),
        ("7. Evaluation", "Accuracy, macro F1, confusion matrix, bias analysis by size, clarity and video."),
        ("8. Explainability", "Attention rollout maps and analysis of confident mistakes."),
        ("9. Comparison", "CLIP zero-shot as a multimodal transformer reference."),
        ("10. Deployment", "Weights on GitHub Releases; this Streamlit app on Streamlit Community Cloud."),
    ]
    for t, d in steps:
        st.markdown(f'<div class="step"><b>{t}</b> · {d}</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown("**Training progress**")
    hist = pd.DataFrame({"train acc": [0.818, 0.972, 0.991, 0.995, 0.997], "val acc": [0.795, 0.782, 0.808, 0.817, 0.813],
                         "val macro F1": [0.741, 0.770, 0.774, 0.789, 0.784]}, index=pd.Index([1, 2, 3, 4, 5], name="epoch"))
    st.line_chart(hist)
    st.caption("Best epoch: 4 (validation macro F1 0.789). The test set was used only once, at the end.")

elif page == "Dataset & Augmentation":
    hero("📁 Dataset & Augmentation", "TrashCan 1.0 (Hong, Fulton & Sattar, 2020), University of Minnesota.")
    kpis([("7,212", "Video frames"), ("12,074", "Object crops"), ("310", "Videos"), ("79 px", "Median object size")])
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Object crops per class and split**")
        t = COUNTS.copy()
        t["total"] = t.sum(1)
        st.dataframe(t, use_container_width=True)
    with c2:
        st.markdown("**Class distribution (all splits)**")
        st.bar_chart(COUNTS.sum(1).rename("objects"))
    st.markdown("""
**Pre-processing**
- Every annotated object is cropped with **10% surrounding context**; boxes smaller than 8 px are skipped.
- Crops are resized to **224 × 224** and normalised with mean = std = 0.5, as in ViT pre-training.
- **Video-level split:** the official TrashCan split shares 113 of its 115 test videos with training, so a new split
  keeps every video in exactly one set. This gives an honest estimate on unseen dives.

**Augmentation (training only)**
- **Random resized crop** (70–100% of the object): varying distance and framing.
- **Horizontal flip:** objects can face either way.
- **Colour jitter** (brightness 0.3, contrast 0.3, saturation 0.2): simulates changing light and turbidity under water.

**Class imbalance:** trash is 49% of the crops, so the loss is class-weighted and macro F1 is the main metric.
""")
    show_fig("fig1_dataset_samples.png", "Training examples: object crops per class")

elif page == "Use Cases & Impact":
    hero("🌍 Use Cases & Impact", "Why automatic litter recognition under water matters.")
    uses = [
        ("🛰️ Automated ROV surveys", "Tag litter in hours of deep-sea video automatically instead of manual review."),
        ("🧹 Clean-up planning", "Map where litter gathers on the sea floor to target clean-up missions."),
        ("📈 Environmental monitoring", "Track the amount of litter over time for research and policy (e.g. UN SDG 14, Life Below Water)."),
        ("🐠 Ecosystem protection", "Separate litter from marine life so robots do not disturb animals."),
        ("🤖 Robot autonomy", "Recognise the ROV's own parts so they are not counted as litter."),
        ("🎓 Education", "A clear example of how Vision Transformers work and where they look."),
    ]
    for i in range(0, len(uses), 3):
        for col, (t, d) in zip(st.columns(3), uses[i:i + 3]):
            col.markdown(f'<div class="card"><h4>{t}</h4>{d}</div>', unsafe_allow_html=True)
        st.write("")
    st.markdown("**Impact:** marine plastic harms wildlife and is very hard to find in the deep sea. "
                "A model that recognises litter with 85.6% accuracy on unseen dives can make large-scale surveys "
                "faster and cheaper, with humans checking the uncertain cases.")

elif page == "About & Limitations":
    hero("ℹ️ About & Limitations", "What the model can and cannot do.")
    st.markdown("""
**Model:** ViT-B/16 (Dosovitskiy et al., 2021), `google/vit-base-patch16-224`, ImageNet pre-trained, all layers fine-tuned
(AdamW, lr 3e-5, 5 epochs, class-weighted loss). 12 transformer layers, 12 attention heads, 85.8 M parameters.

**Limitations**
- Trained only on **deep-sea ROV footage** (mostly the Sea of Japan); bright shallow water, other cameras or objects
  outside the three classes can give confident but wrong answers.
- The model classifies **one object at a time**; it does not find or locate objects in a full scene.
- Accuracy varies between dives (41–100% per video) and is lower for very small objects.
- **Bio** recall is 75.9%: some marine life is mistaken for litter or robot parts.

**References**
- Dosovitskiy et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR.
- Hong, Fulton & Sattar (2020). *TrashCan: A Semantically-Segmented Dataset towards Visual Detection of Marine Debris.*
- Abnar & Zuidema (2020). *Quantifying Attention Flow in Transformers.* ACL.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML.
""")
    st.caption("MAAI901 Deep Learning for Visual Data · University of Wollongong in Dubai")
