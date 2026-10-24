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

st.set_page_config(page_title="ViT Underwater Litter", page_icon="🔬", layout="wide")


def find_file(name):
    for p in (ROOT / name, ROOT / "results" / name, ROOT / "models" / name):
        if p.exists():
            return p
    return None


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
    return pd.read_csv(f, index_col=0) if f else None


def samples():
    files = sorted(set(ROOT.glob("sample__*.jpg")) | set((ROOT / "samples").glob("sample__*.jpg")))
    return {f"{f.name.split('__')[1]} · {f.name.split('__', 2)[2]}": f for f in files}


# ---------------------------------------------------------------- page
st.title("🔬 ViT Underwater Litter")
st.markdown("A **Vision Transformer (ViT-B/16)**, fine-tuned on object crops from the "
            "[TrashCan 1.0](https://conservancy.umn.edu/handle/11299/214865) deep-sea ROV dataset, "
            "classifies an underwater object as **trash**, **bio** (marine life) or **rov** (the robot's own parts).")

tab_cls, tab_perf, tab_about = st.tabs(["🔍 Classify", "📊 Model performance", "ℹ️ About"])
results = load_results()

with st.sidebar:
    st.header("Image")
    src = st.radio("Source", ["Upload an image", "Example test image (true class known)"])
    st.caption("The model classifies **one object**. For a whole scene, use the crop sliders to frame one object.")

with tab_cls:
    img, truth, name = None, None, None
    if src == "Upload an image":
        up = st.file_uploader("Underwater image", type=["jpg", "jpeg", "png", "webp"])
        if up is not None:
            img, name = Image.open(up), up.name
    else:
        ex = samples()
        if ex:
            key = st.selectbox("Example test image", list(ex))
            img, name, truth = Image.open(ex[key]), ex[key].name, key.split(" · ")[0]
        else:
            st.warning("No example images (sample__*.jpg) found in the repository.")

    if img is None:
        st.info("Upload an underwater image, or choose an example test image in the sidebar.")
    else:
        img = img.convert("RGB")
        W, H = img.size
        with st.expander("✂️ Crop to one object (optional)"):
            c1, c2 = st.columns(2)
            x0, x1 = c1.slider("Left / right (%)", 0, 100, (0, 100))
            y0, y1 = c2.slider("Top / bottom (%)", 0, 100, (0, 100))
        if x1 - x0 >= 5 and y1 - y0 >= 5:
            img = img.crop((int(W * x0 / 100), int(H * y0 / 100), int(W * x1 / 100), int(H * y1 / 100)))

        with st.spinner("Loading the ViT (first time only, about a minute)..."):
            model = load_model()
        probs, heat = classify(model, img)
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

        with st.container(border=True):
            if truth is not None:
                ok = truth == NAMES[k]
                st.markdown(f"**Accuracy on this image:** true class **{truth}**, predicted **{NAMES[k]}**: "
                            + ("✅ correct" if ok else "❌ incorrect"))
            else:
                st.markdown("**Accuracy on this image:** not measurable (no true label for an uploaded photo)")
            if results is not None and "ViT fine-tuned" in results.index:
                r = results.loc["ViT fine-tuned"]
                per = " · ".join(f"{c} F1 {r[f'{c} F1']:.1%}" for c in NAMES if f"{c} F1" in r)
                st.markdown(f"**Model accuracy on the test set:** accuracy **{r['accuracy']:.1%}** · "
                            f"macro F1 **{r['macro F1']:.1%}**  \n{per}")

with tab_perf:
    st.subheader("Test-set results")
    st.caption("Objects from deep-sea videos that were never used for training or model selection.")
    if results is not None:
        cols = [c for c in ["accuracy", "macro F1"] + [f"{c} F1" for c in NAMES] if c in results.columns]
        st.dataframe(results[cols].round(3), use_container_width=True)
        if "ViT fine-tuned" in results.index:
            r = results.loc["ViT fine-tuned"]
            m1, m2 = st.columns(2)
            m1.metric("ViT fine-tuned: accuracy", f"{r['accuracy']:.1%}")
            m2.metric("ViT fine-tuned: macro F1", f"{r['macro F1']:.1%}")
    else:
        st.warning("test_results.csv not found in the repository.")

with tab_about:
    st.markdown("""
- **Model:** ViT-B/16 (Dosovitskiy et al., 2021), ImageNet pre-trained, all layers fine-tuned (AdamW, lr 3e-5,
  5 epochs, class-weighted loss). Images are split into 16 × 16 patches and processed by 12 transformer layers.
- **Data:** TrashCan 1.0 (Hong, Fulton & Sattar, 2020). Every labelled object is cut out and labelled
  trash / bio / rov. Whole **videos** are kept in one split, so test objects come from unseen dives.
- **Attention map:** attention rollout (Abnar & Zuidema, 2020) shows which image patches the model relied on.
- **Limits:** trained only on deep-sea ROV footage (mostly the Sea of Japan); bright shallow water, other cameras
  or objects not in the three classes can give confident but wrong answers. The model classifies one object at a
  time and does not locate objects in a scene.

*MAAI901 Deep Learning for Visual Data, University of Wollongong in Dubai.*
""")
