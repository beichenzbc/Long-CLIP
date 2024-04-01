# Long-CLIP Training
To run the training code for Long-CLIP, please follow the following step.

### 1. Prepare CLIP Model
First, download the checkpoints of OpenAI CLIP Model. You can refer to this page https://github.com/openai/CLIP.

Then, you can load the model from CLIP by running the following command. The positional embedding will be stretched from 77 to 248. 
```python
from model import longclip
model, preprocess = longclip.load_from_clip('ViT-B/16', device='cpu')
```
### 2. Prepare ShareGPT4V dataset

First, download all images we used.
- LAION-CC-SBU-558K: [images.zip](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain/blob/main/images.zip)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- WebData: [images](https://drive.google.com/drive/folders/1tCUQ-sq6vdshZVkF0ZeF3K4eztkXJgax?usp=sharing). Only for academic usage.
- SAM: [images](https://ai.meta.com/datasets/segment-anything-downloads/). We only use 000000~000050.tar for now. If you just want to use ShareGPT4V for SFT, you can quickly download 9K images from [here](https://drive.google.com/file/d/1dKumdOKSXtV7lIXdrG7jsIK_z2vZv2gs/view?usp=drive_link). 
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing). We save all files as `.jpg`
- TextVQA: [trainvalimages](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

Then, download the long caption of these image [share-captioner_coco_lcs_sam_1246k_1107.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/blob/main/share-captioner_coco_lcs_sam_1246k_1107.json)


Finally, organize the data as follows in `projects/ShareGPT4V/data`:

```none
ShareGPT4V
├── ...
├── data
|   ├── share-captioner_coco_lcs_sam_1246k_1107.json
│   ├── llava
│   │   ├── llava_pretrain
│   │   │   ├── images
│   ├── coco
│   │   ├── train2017
│   ├── sam
│   │   ├── images
│   ├── gqa
│   │   ├── images
│   ├── ocr_vqa
│   │   ├── images
│   ├── textvqa
│   │   ├── train_images
│   ├── vg
│   │   ├── VG_100K
│   │   ├── VG_100K_2
│   ├── share_textvqa
│   │   ├── images
│   ├── web-celebrity
│   │   ├── images
│   ├── web-landmark
│   │   ├── images
│   ├── wikiart
│   │   ├── images
├── ...
```
Then, change the data root in `sharegpt4v.py`

ShareGPT4V dataset contains 1M (image, long caption) pairs. The short caption used in Primary Component Matching can be obtained by truncating the first sentence of ShareGPT4V since it's usually a summary caption like 'The image showcases ......'. This simple strategy has worked quite well. 
Or, if you want to further improve the quality, you could use a LLM to summarize the long captions and rewrite `sharegpt4v.py`.

### 3. Finetune

Finally, you can run the `train.py` for fine-tuning. Our codebase support both slurm and torch.distributed.launch for distributed data parallel. `train_slurm.sh` is a demo of running `train.py` via slurm.





Recommended Hyper-parameter settings (sorry about the typo of ***learning rate*** in the paper):

| Batch Size | Learning rate | Epochs |  Warm-up iterations | Weight decay |  AdamW&nbsp;β1 |  AdamW&nbsp;β2 |  AdamW&nbsp;ϵ |
| ----: | ----: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2048 | *1e-6* | 6 | 200 | 1e-2 | 0.9 | 0.999 | 1e-8 |