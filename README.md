# 🔬 ViT Underwater Litter

Fine-tuning a **Vision Transformer (ViT-B/16)** to recognise objects seen by deep-sea robots (ROVs) as **trash**, **bio** (marine life) or **rov** (the robot's own parts), using the [TrashCan 1.0](https://conservancy.umn.edu/handle/11299/214865) dataset.

**Live app:** _add your Streamlit link here_

## What is in this repository
| File | Contents |
|---|---|
| `ViT_Underwater_Litter.ipynb` | Full pipeline (Google Colab, T4 GPU): dataset and pre-processing, ViT configuration, frozen-ViT baseline, full fine-tuning, evaluation, bias analysis, attention maps, CLIP zero-shot comparison |
| `app.py` | Streamlit app: upload an image → ViT prediction, class probabilities, attention map, accuracy |
| `test_results.csv` | Test-set results |
| `sample__*.jpg` | Labelled test images used as examples in the app |
| Release `v1.0` | Fine-tuned weights `vit_b16_trashcan_fp16.pt`, downloaded automatically by the app |

## Method in brief
- Every labelled object in TrashCan is cut out (10% context) and labelled with one of the three top-level classes.
- **Video-level split** (70/15/15): the official split shares frames from 113 of its 115 test videos with training, so whole videos are kept together.
- ViT-B/16 (`google/vit-base-patch16-224`), 224 × 224 input, augmentation (random resized crop, flip, colour jitter), AdamW (lr 3e-5, wd 0.05), one-cycle schedule, 5 epochs, class-weighted cross-entropy, best epoch chosen on validation macro F1.

## Run the app locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## References
- Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.
- Hong, J., Fulton, M., & Sattar, J. (2020). TrashCan 1.0: An Instance-Segmentation Labeled Dataset of Trash Observations. https://doi.org/10.13020/g1gx-y834
- Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. ACL.
- Radford, A. et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML.

*MAAI901 Deep Learning for Visual Data, University of Wollongong in Dubai.*
