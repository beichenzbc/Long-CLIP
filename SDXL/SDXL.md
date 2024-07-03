# Long-CLIP-SDXL
To run Long-CLIP-SDXL, please follow the following step.

### 1. Prepare SDXL Model
Download the pre-trained weights of SDXL-base and SDXL-refiner in the following pages: 
[https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
[https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0)

### 2. Prepare the text encoders
Download the pre-trained Long-CLIP-L and OpenCLIP-bigG respectively. Then change the root in `encode_prompt.py`.
[https://huggingface.co/BeichenZhang/LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L)
[https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)
For OpenCLIP-bigG, due to the heavy training cost, we only apply knowledge-preserved stretching of positional embedding without any fine-tuning.

### 3. Start generating images.
Finally, you can run the `sdxl.py` for generating images.

### Notifications
`demo.png` shows some demos of Long-CLIP-SDXL, we can find that the model can now breakthrough the limitation of 77 token('//' marked in red in demo.png) with little loss of quality. However, we found that the quality on generating human face may be affected. You may further fine-tune Long-CLIP-SDXL to achieve a better quality.