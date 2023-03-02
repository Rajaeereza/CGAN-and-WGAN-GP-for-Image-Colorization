# CGAN and WGAN GP for Image Colorization

We implement a conditional Generative adversarial network (CGAN) for the colorizing grayscale images and improve it with Wasserstein Generative adversarial network with gradient penalty (WGAN-GP). We try to solve the problem of gradient vanishing that we face in CGAN training. Although WGAN-GP can solve the problem, it is slow to learn. In addition to these main architectures, we try to colorize images with another CGAN with fewer parameters
in the generator. In all cases, we can see the effect of the power of the discriminator. The fact that the discriminator learns very fast can be considered one of the reasons for the slow learning of the generator.
