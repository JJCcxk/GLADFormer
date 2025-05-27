
# GLADFormer: A Guided Light-Aware Dual-Attention Transformer for Low-Light Image Enhancement

This repository contains the official PyTorch implementation of **GLADFormer**, a novel Transformer-based framework for low-light image enhancement.

## 🌟 Highlights
- **Dual-branch Retinex-inspired Design**: Combines illumination estimation and detail restoration.
- **Local-Global Attention**: Integrates Local Chunked Masked Attention (LCMA) with global token fusion for spatial consistency.
- **Pixel-Aware Refinement**: Uses codebook-guided Pixel-Aware Gated Modulation (PAGM) for fine structure restoration.
- **Contrastive Learning**: Improves illumination representation by separating bright/dark variants.

## 📂 Dataset Preparation

The following datasets are supported:
- LOL-v1
- LOL-v2 (Real & Synthetic)
- SMID
- SID
- ExDark (for object detection)

Please refer to `datasets/README.md` for details on dataset download and structure.

## 🚀 Getting Started

### Prerequisites
- Python ≥ 3.8
- PyTorch ≥ 1.12
- CUDA ≥ 11.3
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Training
```bash
python train.py --config configs/gladformer.yaml
```

### Testing
```bash
python test.py --weights checkpoints/gladformer_best.pth --dataset_path ./datasets/LOL-v1/test
```

### Object Detection (ExDark)
```bash
python detect_with_yolov8.py --enhancer gladformer
```

## 📈 Performance

| Dataset      | PSNR (↑) | SSIM (%) (↑) |
|--------------|----------|---------------|
| LOL-v1       | 24.88    | 83.8          |
| LOL-v2 Real  | 22.36    | 85.3          |
| LOL-v2 Synth | 25.31    | 92.3          |
| SMID         | 29.17    | 81.5          |
| SID          | 24.45    | 67.7          |

## 🔗 Links
- [Project Page](https://github.com/JJCcxk/GLADFormer)

## 📬 Contact

For questions or feedback, feel free to open an issue or contact:
- Yijin Diao (Email: `YijinDiao@163.com`)





