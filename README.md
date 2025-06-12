# WGAN-GP for Weather Downscaling with Extreme Event Detection

This repository implements a **Wasserstein GAN with Gradient Penalty (WGAN-GP)** using a U-Net-style (no skip-connections) generator and a patch-based discriminator, designed for high-resolution downscaling of weather and climate data, with a special emphasis on **extreme wind gusts**.

##  Features

-  **U-Net-based Generator** with stochastic noise input (`z`)
-  **Dual-head Discriminator**: 
  - Patch realism map
  - Binary classifier for extreme event detection
-  **Tail-weighted & quantile-aware loss functions**
-  Support for:
  - MinMaxScaler-based normalization
  - SSIM + L1 + adversarial + BCE loss fusion
  - Gradient penalty (WGAN-GP)

---

## Directory Structure 
`````
main.py                   # Entry point
├── models/
│   ├── generator.py      # Generator architecture
│   └── discriminator.py  # Discriminator architecture
├── losses/
│   └── loss_functions.py # Loss definitions
├── data/
│   └── preprocessing.py  # Normalization and batching
├── training/
│   └── train_loop.py     # Training logic
├── utils/
│   └── dropout.py        # Custom always-on Dropout layer
`````

---

## Coming soon:
- **Evaluation scripts**
- **Extreme event detection accuracy**
