## Reproducing results for the modern equivariant ViT

> Download patch Camelyon here:
https://drive.google.com/file/d/1THSEUCO3zg74NKf_eb3ysKiiq2182iMH/view?usp=sharing

---

For the experiment with modern ViT using patches:

```bash
python eq_modern_vit.py --modern_vit
```

For the experiment with the modern ViT using a cnn:

```bash
python eq_modern_vit.py --modern_vit_w_cnn
```

To run their model:

```bash
python eq_modern_vit.py
```

Run ablation:
```bash
./experiment.sh
```