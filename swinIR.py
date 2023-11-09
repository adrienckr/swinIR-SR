from mmagic.apis import MMagicInferencer
import time
config = 'configs/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-ost.py'
checkpoint = 'https://download.openmmlab.com/mmediting/swinir/swinir_gan-x4s64w8d9e240_8xb4-lr1e-4-600k_df2k-os-9f1599b5.pth'
img_path = 'flyy.png'
start_time = time.time()
editor = MMagicInferencer('esrgan', model_config=config, model_ckpt=checkpoint)

output = editor.infer(img=img_path,result_out_dir='upscaled1.png')

end_time = time.time()
print(f"Inference time: {end_time - start_time:.2f} seconds")
