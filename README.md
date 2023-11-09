
# 📷 Image Denoising with MMagic's DeNoise Model 🌟

This Python script demonstrates how to use the MMagic API to remove noise from images and enhance their quality using the DeNoise model. Simply provide an input image, and the script will generate a denoised output! 🧹🖼️
## Prerequisites 🛠️

Before you get started, make sure you have the following dependencies installed:

- Python 3.8 (or a compatible version) 🐍
- Miniconda ⚙️

## Installation Steps 📦

Follow these simple steps to set up your environment for using the MMagic Stable Diffusion model:

1. **Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) on your system if you haven't already.**

2. **Create a Conda environment and activate it using the following commands:**

   ```bash
   conda create --name mmagic python=3.8 -y
   conda activate mmagic
   ```

3. **Check your CUDA version by running the command:**

   ```bash
   nvcc --version
   ```

4. **Install PyTorch by referring to the [official PyTorch documentation](https://pytorch.org/). Select the appropriate configuration based on your system and follow the installation instructions.**

   ![PyTorch Installation]

5. **Install the `MMCV` library using `MIM`, a package manager for AI and machine learning dependencies. Run the following commands:**

   ```bash
   pip install -U openmim
   mim install 'mmcv>=2.0.0'
   ```

6. **Install `mmengine` from the GitHub repository:**

   ```bash
   pip install git+https://github.com/open-mmlab/mmengine.git
   ```

7. **Install the `mmagic` toolbox in editable mode using the following command:**

   ```bash
   git clone https://github.com/open-mmlab/mmagic.git
   cd mmagic
   pip3 install -e . -v
   ```

   The `-e .` flag is used to install the Python package in editable mode, meaning that any changes made to the source code will be reflected in the installed package.

8. **Additionally, install `accelerate` for faster and less memory-intensive model loading:**

   ```bash
   pip install accelerate
   ```

9. **Run the script:**

   ```bash
   python app.py
   ```

4. **Get the required model configuration and checkpoint from [Models-and-config.txt](Models-and-config.txt).**

5. **Enter the input and output image path.**

   ```bash
   Enter the input image path: input_image.png
   Enter the output image path: output_image.png
   ```

You will get an enhanced quality of your image by increasing the resolution of the image with MMagic's SwinIR Image Super-Resolution model! 📷✨

Make sure to provide the correct paths for your input and output images to enjoy beautifully denoised results. 🧼🌈
