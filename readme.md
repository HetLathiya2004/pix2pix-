**Pix2Pix Implementation**

This repository provides an implementation of the Pix2Pix model for image-to-image translation tasks. Pix2Pix leverages Conditional GANs (Generative Adversarial Networks) to perform tasks like turning sketches into photos, day-to-night translation, and much more.

**Overview**


Pix2Pix is a supervised GAN model that uses paired datasets to learn a mapping from input images (e.g., sketches) to output images (e.g., photos). This implementation supports:

Image-to-image tasks
Model training with visualization
Custom dataset integration.


The **V1 file** includes Pix2Pix model training with the addition of Laplacian Loss to enhance edge details and improve image generation quality.


The **main.py file** includes Pix2Pix model training where the generator is updated 3 times more than the discriminator, incorporating Laplacian Loss for better edge preservation. Additionally, it contains the implementation of WGAN-GP (Wasserstein GAN with Gradient Penalty).


