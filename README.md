# 🌀 ∂-NO: Fractional is Better — Learnable Derivative Orders in Neural Operator Learning

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/Conferences/2026)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)

**[Paper](https://openreview.net/forum?id=V4FDM692AZ) | [arXiv](https://arxiv.org/) | [Slides](https://icml.cc/) | [Poster](https://icml.cc/)**

## 🎯 TL;DR

**∂-NO (del-NO)** is a drop-in augmentation that injects **learnable fractional derivative features** into any neural operator backbone. Both the derivative orders $\beta_k$ and feature scales $s_k$ are jointly optimized with network weights via a differentiable Grünwald–Letnikov stencil. **The central finding:** the MSE-optimal derivative order $\beta^*$ is *strictly less* than the PDE order $m$, with the gap closed-form predicted by a spectral bias–variance tradeoff (Theorem 5.6).

<p align="center">
  <img src="figures/architecture.png" width="88%">
  <br>
  <em>∂-NO architecture: input <code>(x, u)</code> augmented with learnable fractional derivative features <code>D^β u</code> before any unchanged neural-operator backbone (FNO, Transolver, LocalNO, CNO, PINO, …)</em>
</p>

## ✨ Why ∂-NO?

- **🔌 Drop-in**: Wraps any neural-operator backbone (FNO, TFNO, LocalNO, Transolver, CNO, PINO) **without architectural modification**
- **📐 Theory-grounded**: Theorem 5.6 predicts and explains $\beta^\* < m$ via a closed-form bias–variance tradeoff
- **📊 Universally tested**: Improves  across 5 PDEs × 4 backbones × 2 noise regimes (40/40 in Table 1)
- **🌀 Differentiable**: Learnable orders via Grünwald–Letnikov stencil; gradients flow through $\partial_\beta$
- **⚡ Light**: 25–40% wall-clock overhead, no extra hyperparameter search
- **🎯 Robust**: 120/120 stress-test wins under both Gaussian and uniform noise

## 🚀 Quick Start

∂-NO is **architecture-agnostic preprocessing**. In one line of pseudocode:

```python
# ∂-NO augmentation, wraps any backbone NO_θ
x_aug = torch.cat([x, u, *[s_k * D_beta_k(u) for k in range(K)]], dim=-1)
y     = NO_theta(x_aug)
```

where each `D_beta_k` is a Grünwald–Letnikov fractional derivative with a *learnable* order $\beta_k$ and *learnable* scale $s_k$, optimized end-to-end alongside the backbone weights $\theta$.

All notebooks include the complete ∂-NO implementation inline — no complex dependencies, no deep frameworks needed beyond what your backbone already requires.

## 📚 Interactive Notebooks

| Notebook | Backbone | Demonstrates | Runtime |
|----------|----------|--------------|---------|
| [**1. Quick Demo**](notebooks/1_delNO_Demo_Burgers.ipynb) | FNO | • Burgers ν=0.1 baseline vs ∂-FNO<br>• Learnable β trajectory<br>• Single-channel theory validation | 5–10 min |
| [**2. Theory Validation**](notebooks/2_delNO_Theory_Sweeps.ipynb) | FNO | • Three sweeps: σ ↑ ⇒ β* ↓; n ↑ ⇒ β* ↑; k ↑ ⇒ β* ↓<br>• Reproduces Figures 2 and 3<br>• Closed-form δ* tracking | 15–30 min |
| [**3. Multi-Backbone**](notebooks/3_delNO_Multi_Backbone.ipynb) | FNO / Transolver / LocalNO | • Reproduces main Table 1 (5 PDEs × 4 backbones × 2 noise)<br>• 40/40 universal improvement | 1–2 h |
| [**4. ∂-CNO Differential Bias**](notebooks/4_delCNO_Raonic_Benchmarks.ipynb) | CNO | • Seven Raonic et al. benchmarks<br>• 14/14 ∂-CNO wins, peaks 55–65% (Poisson, Wave) | 1–2 h |

### Running the Notebooks

#### Option 1: Google Colab (Recommended for quick start)
Each notebook includes a Colab badge — click to open in a hosted GPU runtime, no local setup.

#### Option 2: Local Setup
```bash
# Install base requirements
pip install jupyter numpy pandas matplotlib torch

# For Transolver / CNO comparisons (THUML library)
pip install -e git+https://github.com/thuml/Neural-Solver-Library.git#egg=neuralop

# For 3D compressible NS (notebook 6) — requires CUDA GPU
pip install -r requirements_3d.txt
```

### Launch
```bash
jupyter notebook notebooks/1_delNO_Demo_Burgers.ipynb
```

All notebooks are self-contained with the ∂-NO implementation included inline. Start with notebook 1 for a 5-minute introduction, or jump directly to your domain of interest.


## 🔧 Data Preparation

### Required Datasets

Place these files in `data/` directory:

| Dataset | Files | Source & Credit |
|---------|-------|-----------------|
| **Burgers (1D)** | `burgers_R10.mat`, `burgers_shock.mat` | [Li et al. 2020 (FNO)](https://github.com/neuraloperator/neuraloperator) |
| **Darcy Flow (2D)** | `Darcy_241.mat` | [Li et al. 2020 (FNO)](https://github.com/neuraloperator/neuraloperator) |
| **Navier–Stokes (2D)** | `NavierStokes_V1e-5.mat` | [Li et al. 2020 (FNO)](https://github.com/neuraloperator/neuraloperator) |
| **KdV (1D, in-house)** | `KdV_smoothBurgersIC.mat` | Generated by direct numerical integration with smooth-Burgers initial conditions; script in [`data/generate_kdv.py`](data/generate_kdv.py) |
| **Raonic CNO Benchmarks** | Poisson, NS-Shear, Wave, Allen–Cahn, Transp.Cont., Darcy, Airfoil | [Raonic et al. 2023 (CNO)](https://github.com/bogdanraonic3/ConvolutionalNeuralOperator) |
| **3D Compressible NS** | `cns3d_64x64x64.h5` | Generated via spectral solver; see [`data/generate_cns3d.py`](data/generate_cns3d.py) |

```bash
# Quick setup
mkdir -p data/
cd data/

# Download Li et al. 2020 datasets
wget https://drive.google.com/uc?id=<burgers_R10_id> -O burgers_R10.mat
wget https://drive.google.com/uc?id=<darcy_241_id>   -O Darcy_241.mat
wget https://drive.google.com/uc?id=<ns_1e5_id>      -O NavierStokes_V1e-5.mat

# Raonic et al. 2023 CNO benchmarks (~6 GB)
# See https://github.com/bogdanraonic3/ConvolutionalNeuralOperator for download instructions

# Generate KdV and 3D CNS data locally
python generate_kdv.py
python generate_cns3d.py

cd ..
```

### Auto-Generated Data

- **KdV (Notebook 1 & 3)**: Generated by direct numerical integration from smooth-Burgers initial conditions
- **3D CNS (Notebook 6)**: Spectral solver, $64\times 64\times 64$ grid, 10-step horizon

All third-party datasets are used in accordance with their respective licenses for academic research.

## 💡 Tips

- **GL stencil length**: Default $K_{\mathrm{GL}} = 16$ balances accuracy and compute; $K_{\mathrm{GL}}=8$ is acceptable, below 8 degrades fractional fidelity (see Appendix L.9)
- **β learning rate**: Use $10\times$ the backbone learning rate (β parameters sit at the input level, gradients propagate back through the full network and attenuate)
- **Channel count**: $K=2$ for 1D problems, $K=5$ for 2D (4 directional + 1 mixed partial), $K=3$ for KdV (matched to $m=3$)
- **Initialization**: $\beta = 1.0, 2.0$ (matching integer orders); add $\beta_3 = 3.0$ for KdV
- **Resolution under memory constraints**: Coarser grids actually *amplify* ∂-NO's relative advantage (Appendix L.4)

## 📈 Key Insights Demonstrated

1. **Sub-physical regularization**: Optimal $\beta^* < m$ strictly in every practical regime; the recovery threshold $n^* = \sigma^2 \exp(A_c)$ is exponentially unreachable
2. **Theory–experiment match**: Closed-form $\delta^* = (A_c - \ln(n/\sigma^2)) / (2\ln(\xi_c \xi_{\max}))$ predicts three monotonicities, all confirmed empirically
3. **Differential inductive bias is what matters**: Matched-parameter controls (Random-FNO, Wider-FNO) confirm gains come from GL derivative structure, not added capacity
4. **Emergent symmetry**: Learned $\beta_x, \beta_y$ converge to within $\Delta < 0.035$ on isotropic PDEs without being prescribed
5. **Complementary to PI losses**: Adding ∂-features to PINO improves PINO in 8/8 settings — input-side and loss-side encodings are orthogonal channels

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `nan` losses early in training | Check that GL stencil omits $h^{-\beta}$ factor (or reduce learning rate for β) |
| β stuck at initialization | Verify β learning rate is $10\times$ backbone LR; check β gradient magnitudes |
| GL stencil dominates compute | Reduce $K_{\mathrm{GL}}$ to 8; trade-off documented in Appendix L.9 |
| 2D / 3D feature explosion | Ensure mixed-partial features use shared scaling factor (one $s$, two $\beta$s) |
| Backbone import errors | THUML Neural-Solver-Library required for Transolver / TFNO / LocalNO |
| 3D CNS out-of-memory | Use gradient checkpointing in Notebook 6, or reduce batch size to 1 |

## 📝 Notes

- The ∂-NO augmentation is **architecture-agnostic** — the same code wraps FNO, TFNO, LocalNO, Transolver, CNO, and PINO
- Code is intentionally kept inline in notebooks to showcase implementation simplicity
- The Grünwald–Letnikov stencil is preferred over Weyl (FFT) on non-periodic domains to avoid Gibbs ringing
- Implementation omits the $h^{-\beta}$ normalization (absorbed by learnable scales $s_k$) for training stability

## 📖 Citation

If you use ∂-NO, please cite:

```bibtex
@inproceedings{mehouachi2026fractional,
  title={Fractional is Better: Learnable Derivative Orders in
         Neural Operator Learning},
  author={Mehouachi, Fares B. and Jabari, Saif Eddin},
  booktitle={Proceedings of the 43rd International Conference on
             Machine Learning},
  year={2026}
}
```

## 🤝 Authors

- **Fares B. Mehouachi** — NYU Abu Dhabi ([fares.mehouachi@nyu.edu](mailto:fares.mehouachi@nyu.edu))
- **Saif Eddin Jabari** — NYU Tandon School of Engineering & NYU Abu Dhabi ([sej7@nyu.edu](mailto:sej7@nyu.edu))

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work was supported by the NYUAD Center for Interacting Urban Networks (CITIES), funded by Tamkeen under the NYUAD Research Institute Award CG001. The views expressed in this article are those of the authors and do not reflect the opinions of CITIES or their funding agencies.

We thank the authors of the [Neural-Solver-Library](https://github.com/thuml/Neural-Solver-Library) (THUML) for the FNO / TFNO / LocalNO / Transolver implementations, and the authors of [Convolutional Neural Operator](https://github.com/bogdanraonic3/ConvolutionalNeuralOperator) for the CNO benchmarks.

---

<p align="center">
<b>Found this useful? Please ⭐ star us on GitHub!</b>
</p>
