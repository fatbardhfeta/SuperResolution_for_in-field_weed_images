# Image Super-Resolution Models

This repository contains three different image super-resolution models: SRCNN, HAT, and real-SRGAN. Each model has its own dedicated directory with a README file providing specific instructions for setup and usage. It's important to run each model in its specific environment due to the differences in architecture, training data, and requirements.

## Models



1. **SRCNN (Super-Resolution Convolutional Neural Network)**

   - [SRCNN README](./srcnn/README.md)
   - The SRCNN model is one of the foundational models in image super-resolution. It has been widely used as a benchmark and was the baseline model for many years. While it may not be the current state of the art, it serves as a reliable and simple neural network-based approach for super-resolution tasks.

2. **HAT (Highway and Attention Network)**

   - [HAT README](./hat/README.md)
   - The HAT model is recognized as one of the best-performing models for super-resolution tasks according to [Papers with Code](https://paperswithcode.com). Its innovative architecture, which includes attention mechanisms, makes it stand out in terms of both quality and efficiency.

3. **real-SRGAN (Super-Resolution Generative Adversarial Network)**

   - [real-SRGAN README](./real-srgan/README.md)
   - The real-SRGAN model represents the current state of the art in GAN-based generative models for super-resolution. It has achieved remarkable results in generating high-quality super-resolved images and is a leading model in the field.


These models represent significant advancements in the field of image super-resolution. HAT is recognized for its performance on Papers with Code, real-SRGAN is the state of the art for GAN-based generative models, and SRCNN has a strong historical significance as a foundational model in the super-resolution domain.


## Running the Models (tips)

- To run a specific model, navigate to its respective directory and follow the instructions provided in the corresponding README file. Ensure that you have the required dependencies, datasets, and model weights ready before proceeding.

- We recommend maintaining separate virtual environments for each model to avoid conflicts between dependencies and ensure reproducibility. Running the models in their dedicated environments will help you achieve the best results and streamline the development process.


If you have any **questions** or **encounter issues** while working with these models, please refer to the model-specific README files for troubleshooting and support or write an email at: **fatbardh.feta@tum.de**


