# PSR-SCI: Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement

[![Paper](https://img.shields.io/badge/Paper-ICLR%202025-blue)](https://openreview.net/pdf?id=Q150eWkQ4I)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-Google%20Drive-orange)](https://drive.google.com/drive/folders/1eR-tN4v5WWHmsrhTmqAJuk8LvJUwpgE_?usp=drive_link)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/SMARK2022/PSR-SCI/blob/main/LICENSE)


**[ICLR 2025]** Official code implementation for **PSR-SCI: Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement**.

[Haijin Zeng](https://scholar.google.com/scholar?start=20&q=haijin+zeng)¹∗,
[Benteng Sun](https://smark2022.github.io/)²∗,
Yongyong Chen²†,
Jingyong Su²†,
Yong Xu² <br>
¹ Telin, Ghent University, ² Harbin Institute of Technology (Shenzhen) <br>
(*Equal contribution, †Corresponding author)

## TABLE OF CONTENTS
1. [News](#news)
2. [Highlights](#highlights)
3. [Visual Results](#visual-results)
4. [Installation](#installation)
5. [Datasets and Pre-trained Models](#datasets-and-pre-trained-models)
6. [Quick Start: Inference](#quick-start-inference)
7. [Training](#training)
8. [Evaluation](#evaluation)
9. [Citation](#citation)
10. [Acknowledgement](#acknowledgement)
11. [License](#license)

---

## News
- **[Oct 2025]** The code, pre-trained models, and results for PSR-SCI are publicly released.
- **[Apr 2025]** PSR-SCI has been accepted by **ICLR 2025**. 🚀

---

## Highlights

<p align="center" width="80%">
<img src="assets/PSR-SCI.png" alt="SOTA Comparison" style="width: 80%; min-width: 300px; display: block; margin: auto;">
</p>

1.  **SOTA Performance:** PSR-SCI achieves state-of-the-art reconstruction performance on standard benchmarks (KAIST) and strong zero-shot generalization (NTIRE, ICVL, Harvard), significantly outperforming previous deterministic and generative methods in both PSNR/SSIM and perceptual quality.
2.  **Novel PSR-SCI Framework:** We propose a **Predict-and-unmixing-driven-Subspace-Refine** framework. It first uses a cost-effective predictor for a coarse estimate, then focuses a diffusion model on refining only the high-frequency details.
3.  **Efficient Diffusion on MSI:** We introduce a **Unmixing-driven Reversible Spectral Embedding (URSe)** module. This key component decomposes high-dimensional MSIs into low-dimensional subspace images and spectral coefficients, allowing us to adapt powerful, pre-trained **RGB diffusion models** for MSI refinement. This drastically reduces computational costs and the need for large-scale MSI training data.
4.  **High-Fidelity Guidance:** A novel high-dimensional guidance mechanism with imaging consistency is designed to steer the diffusion sampling process, ensuring the refined output remains faithful to the physical SCI measurement.
5.  **Fast Inference:** Our subspace-based refinement strategy is highly efficient. PSR-SCI runs in just **8.9 seconds** (for 50 steps), which is ~10x faster than prior diffusion-based methods like DiffSCI (84.5s).

---

## Visual Results

### Results on Simulation Dataset (KAIST)
PSR-SCI recovers finer textures and facial details with fewer artifacts compared to other SOTA methods.
<p align="center" width="100%">
<img src="assets/teaser.png" alt="KAIST Results" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

### Results on Real-World Dataset
Our method successfully reconstructs the "star" shape with sharp edges and a clean surface, while other methods produce blurred or incomplete structures.
<p align="center" width="100%">
<img src="assets/real_dataset.png" alt="Real Data Results" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

### Results on Zero-shot Dataset

PSR-SCI demonstrates strong zero-shot generalization, accurately reconstructing fine details and sharp edges on unseen real-world data. Competing methods yield blurred or distorted shapes, while PSR-SCI preserves structural integrity and surface clarity.

<p align="center" width="100%">
<img src="assets/zero_shot.jpg" alt="Zero-shot Data Results" style="width: 100%; min-width: 300px; display: block; margin: auto;">
</p>

---

## Installation

1.  Clone this repository:
    ```bash
    git clone https://github.com/SMARK2022/PSR-SCI.git
    cd PSR-SCI
    ```

2.  Create and activate a virtual environment (e.g., using conda):
    ```bash
    conda create -n psrsci python=3.9
    conda activate psrsci
    ```

3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

---

## Datasets and Pre-trained Models

### 1. Download Datasets

Our framework is trained and tested on standard SCI datasets (CAVE, KAIST, etc.). Please follow the download instructions from the [MST repository](https://github.com/caiyuanhao1998/MST) to download and prepare the datasets.

Place the datasets in the `packages/MST/datasets/` directory, following the structure outlined in their README.

### 2. Download Pre-trained Models

Our model relies on several pre-trained components.

#### **PSR-SCI Weights (Ours)**
Download our trained weights for the URSe (SeVAE), VAE, and ControlNet.
- **Download Link:** [Google Drive](https://drive.google.com/drive/folders/1eR-tN4v5WWHmsrhTmqAJuk8LvJUwpgE_?usp=drive_link)
- **Destination:** Place the `.pt` files in the main `weights/` directory.

```

└── 📁weights
├── controlnet_sample0160000.pt
├── model_SeVAE_hf3_endecoder_c21_bu2_c9_DConvWoBN_resca_silu_2024-09-05_psnr49.5199.pt
└── vae_sample0012000.pt

```

#### **Dependency Weights (DiffBIR)**
Download the pre-trained weights for the underlying DiffBIR framework.
- `v2.pth`: [Download from Hugging Face](https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth)
- `v2-1_512-ema-pruned.ckpt`: [Download from Stability AI](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt)
- **Destination:** Place these files in `packages/DiffBIR/weights/`.

```

└── 📁PSR-SCI
└── 📁packages
└── 📁DiffBIR
└── 📁weights
├── v2-1_512-ema-pruned.ckpt
└── v2.pth

```

#### **Dependency Weights (MST & DAUHST)**
Download the pre-trained predictor models from the [MST repository](https://github.com/caiyuanhao1998/MST).
- **Download:** Follow the links in the MST repo for `mst_l.pth` and `dauhst_3stg.pth`.
- **Destination:** Place them in `packages/MST/simulation/test_code/model_zoo/`.

```

└── 📁PSR-SCI
└── 📁packages
└── 📁MST
└── 📁simulation
└── 📁test_code
└── 📁model_zoo
├── 📁dauhst_3stg
│   └── dauhst_3stg.pth
└── 📁mst
└── mst_l.pth

```

---

## Quick Start: Inference

We provide easy-to-use Jupyter notebooks for running inference on both simulated and real-world data.

1.  **Inference on Simulated Data (KAIST):**
    Open and run `tools/simulation_inferance_demo.ipynb` to see the full pipeline for reconstructing simulated measurements.

2.  **Inference on Real-World Data:**
    Open and run `tools/real_inferance_demo.ipynb` to test PSR-SCI on real-world SCI captures.

You can also find our pre-computed `.mat` results for the KAIST dataset in the `results/` folder.

---

## Training

The training process involves two main stages. (Note: A unified training script will be released soon. For now, the components can be trained as described below.)

1.  **Stage 1: Train the Unmixing-driven Reversible Spectral Embedding (URSe)**
    The URSe module (referred to as SeVAE in the code) is trained first to learn the spectral embedding.
    ```bash
    python tools/train_Encoder_hf.py
    ```

2.  **Stage 2: Train the Subspace Diffusion Refinement Model**
    After freezing the URSe, we train the VAE and ControlNet to refine the high-frequency subspace images. This code is based on the DiffBIR package.
    ```bash
    # Configure your dataset paths in the .yaml file
    # (e.g., packages/DiffBIR/configs/train/train_stage2.yaml)
    accelerate launch train_vae_new.py --config configs/train/train_vae.yaml
    accelerate launch train_stage2.py --config configs/train/train_stage2.yaml
    ```

<!-- ---

## Evaluation

To evaluate the reconstruction results, you can use the quality assessment scripts provided in the `MST` package.

1.  Run the inference notebooks to generate your `.mat` reconstruction files.
2.  Use the MATLAB scripts in `packages/MST/simulation/test_code/Quality_Metrics/` to calculate **PSNR**, **SSIM**, and **CC** metrics.
    ```matlab
    % Example in MATLAB
    % Add the path to the metric functions
    addpath('packages/MST/simulation/test_code/Quality_Metrics/');
    
    % Load ground truth and your reconstruction
    load('path/to/ground_truth.mat'); % e.g., 'gt'
    load('path/to/your_result.mat');  % e.g., 'recon'
    
    % Calculate metrics
    [psnr_val, ssim_val, ~] = quality_assessment(gt, recon, 256, 256);
    cc_val = CC(gt, recon);
    
    fprintf('PSNR: %.4f, SSIM: %.4f, CC: %.4f\n', psnr_val, ssim_val, cc_val);
    ``` -->

---

## Citation
If you find our work useful for your research, please consider citing our paper:
```bibtex
@inproceedings{zeng2025spectral,
  title={Spectral Compressive Imaging via Unmixing-driven Subspace Diffusion Refinement},
  author={Zeng, Haijin and Sun, Benteng and Chen, Yongyong and Su, Jingyong and Xu, Yong},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
````

-----

## Acknowledgement

This project is built upon the excellent work of several open-source projects. We extend our sincere thanks to their authors.

  - **[DiffBIR](https://github.com/XPixelGroup/DiffBIR)**: For the powerful diffusion-based image restoration framework.
  - **[MST](https://github.com/caiyuanhao1998/MST)**: For the transformer-based SCI reconstruction methods and the comprehensive dataset/evaluation codebase.

-----

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
