# CGAN and WGAN-GP for Image Colorization

---

## Overview

Image colorization — transforming grayscale images into plausible color images — is a challenging generative task that requires learning complex mappings between pixel intensity and color distributions. This project implements and compares two GAN-based approaches for automatic image colorization: a Conditional GAN (CGAN) and a Wasserstein GAN with Gradient Penalty (WGAN-GP).

Both architectures use U-Net as the generator and PatchGAN as the discriminator, operating in the L\*a\*b color space — where the L channel (luminance) serves as input and the network predicts the a and b color channels.

---

## Architectures

### CGAN (Conditional GAN)

The generator is a **U-Net** with 8 encoder blocks and 7 decoder blocks connected via skip connections. Input images of shape (1, 256, 256) are progressively downsampled to a bottleneck of (512, 1, 1) then reconstructed to (2, 256, 256) — the predicted a and b color channels.

The discriminator is a **PatchGAN** that classifies overlapping 70×70 patches as real or fake, producing a (30×30) output map rather than a single scalar. This encourages local texture realism.

The objective combines adversarial loss with an L1 reconstruction term (λ=100):

```
G* = argmin_G max_D V(G,D) + λ · L1
```

### WGAN-GP (Wasserstein GAN with Gradient Penalty)

WGAN-GP replaces the binary discriminator with a **critic** that outputs a score rather than a probability, minimising the Wasserstein distance between real and generated distributions. This directly addresses the vanishing gradient problem observed in CGAN training.

The gradient penalty enforces the Lipschitz constraint on the critic:

```
L = E[D(x̃)] - E[D(x)] + λ · E[(||∇D(x̂)||₂ - 1)²]
```

where x̂ is an interpolation between real and generated samples and λ=10. Batch normalisation is replaced with instance normalisation in the critic to avoid correlation artifacts during gradient penalty computation.

---

## Experiments

Four experimental configurations were evaluated:

| Experiment | Generator params | Dataset | Training |
|---|---|---|---|
| CGAN — ImageNet | 54.4M | ImageNet mini-100 (4,750 train) | 100 epochs |
| CGAN — COCO | 54.4M | MS-COCO subset (7,650 train) | 100 epochs |
| CGAN Mini — COCO | 6.3M | MS-COCO subset (7,650 train) | 100 + 200 epochs |
| WGAN-GP — COCO | 54.4M | MS-COCO subset (7,650 train) | 100 epochs |

All images resized to 256×256. Data augmented with random horizontal flip. Batch size 16 for COCO experiments. Both generator and discriminator optimised with Adam (lr=0.0002, betas=(0.5, 0.999)) for CGAN, and Adam (lr=0.0001, betas=(0, 0.9)) for WGAN-GP.

---

## Results and findings

**Dataset quality matters significantly.** CGAN trained on ImageNet mini produced unsatisfying colorizations and slow convergence. Switching to MS-COCO — a larger, more diverse dataset with better resolution — substantially improved generator convergence speed and visual quality.

**Vanishing gradient is a fundamental limitation of CGAN.** After approximately epoch 60, generator loss plateaus consistently across COCO experiments. The discriminator learns too quickly, leaving the generator unable to improve. This is the core motivation for WGAN-GP.

**Reducing model capacity does not solve the generator-discriminator imbalance.** The CGAN Mini architecture (6.3M vs 54.4M generator parameters) resolved the overfitting observed in the full CGAN but introduced a new problem: the generator became too weak relative to the discriminator, causing generator loss to increase rather than converge.

**WGAN-GP resolves vanishing gradients but converges slowly.** The Wasserstein critic continues to provide meaningful gradients beyond epoch 60, where CGAN stagnates. However, WGAN-GP requires significantly more epochs to produce visually satisfying results — 100 epochs were insufficient for good colorization quality.

**Human evaluation confirmed CGAN's perceptual quality on in-distribution images.** A perceptual study with 30 participants (20 image pairs, 5-second viewing time) found that CGAN-COCO colorizations fooled participants 7.5% of the time on COCO images, but performed notably worse on out-of-distribution images — revealing that the model learned dataset-specific color statistics rather than generalising.

---

## Key questions addressed

- Can a CGAN with U-Net and PatchGAN produce perceptually convincing colorizations?
- Does WGAN-GP effectively resolve the vanishing gradient problem observed in CGAN?
- How does dataset choice (ImageNet vs MS-COCO) affect colorization quality?
- What is the effect of reducing generator capacity on the generator-discriminator training dynamic?

---

## Technical skills

| Area | Details |
|---|---|
| Generative models | CGAN · WGAN-GP · adversarial training · Wasserstein distance |
| Architecture design | U-Net · PatchGAN · skip connections · encoder-decoder · gradient penalty |
| Color spaces | L\*a\*b color space · luminance-chrominance separation |
| Training techniques | L1 regularisation · instance normalisation · dropout · Adam optimisation |
| Scientific Python | PyTorch · torchvision · MS-COCO · ImageNet · Matplotlib |
| Evaluation | Perceptual human evaluation · loss curve analysis · qualitative comparison |

---

## References

- Isola P, Zhu JY, Zhou T, Efros AA. *Image-to-Image Translation with Conditional Adversarial Networks.* CVPR, 2017.
- Arjovsky M, Chintala S, Bottou L. *Wasserstein GAN.* arXiv:1701.07875, 2017.
- Gulrajani I, Ahmed F, Arjovsky M, Dumoulin V, Courville A. *Improved Training of Wasserstein GANs.* arXiv:1704.00028, 2017.
- Goodfellow IJ et al. *Generative Adversarial Nets.* NeurIPS, 2014.
- Nazeri K, Ng E, Ebrahimi M. *Image Colorization using Generative Adversarial Networks.* MVIP, 2020.

---

*Course project · Deep Learning · University of Padova · 2022–2023*  
*Reza Rajaee · Sarvenaz Babakhani*
