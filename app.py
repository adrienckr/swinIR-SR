from mmagic.apis import MMagicInferencer
import time
import torch

# Model configuration and checkpoint
config = 'configs/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py'
checkpoint = 'https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-9f1599b5.pth'

# Get input and output image paths from the user
input_img_path = input("Enter the input image path: ")
output_img_path = input("Enter the output image path: ")

# Start the timer
start_time = time.time()

# Create the MMagicInferencer
editor = MMagicInferencer('esrgan', model_config=config, model_ckpt=checkpoint)

# Perform image super resolution and save the output image
output = editor.infer(img=input_img_path, result_out_dir=output_img_path)

# Stop the timer and print the inference time
end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")
