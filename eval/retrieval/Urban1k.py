import json
import cv2
from PIL import Image
import sys
sys.path.append('../..')
from model import longclip
import torch
import torch.utils.data as data
import os
import numpy as np

image_root = 'data/Urban1k/image/'
caption_root = 'data/Urban1k/caption/'

class local_dataset(data.Dataset):
    def __init__(self):
        self.image_root = image_root
        self.caption_root = caption_root
        self.total_image = os.listdir(image_root)
        self.total_caption = os.listdir(caption_root)
        model, preprocess = longclip.load("../../checkpoints/longclip-B.pt", device='cuda')
    def __len__(self):
        return len(self.total_caption)

    def __getitem__(self, index):
        caption_name = self.total_caption[index]
        image_name = self.total_caption[index][:-4] + '.jpg'
        image = Image.open(self.image_root + image_name)
        f=open(self.caption_root + caption_name)
        caption = f.readlines()[0]
        
        return image, caption

if __name__ == '__main__':
    dataset = local_dataset()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = longclip.load("../../checkpoints/longclip-B.pt", device=device)
    model.eval()
    print("model done!")
    
    img_feature_list = []
    text_list_1 = []
    text_list_2 = []
    text_list = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (image, caption) in enumerate(dataset):
            text_list.append(caption)

        text_feature = longclip.tokenize(text_list, truncate=True).to(device)
        text_feature = model.encode_text(text_feature)
        text_feature /= text_feature.norm(dim=-1, keepdim=True)
        
        for i, (image, caption) in enumerate(dataset):            
            image = preprocess(image).unsqueeze(0).to(device)
            img_feature = model.encode_image(image)
            img_feature_list.append(img_feature)
            
        image_embeds = torch.cat(img_feature_list, dim=0)
        image_embeds /= image_embeds.norm(dim=-1, keepdim=True)
        
        print("text 2 image")
        i = 0
        correct = 0
        total = 0
        for i in range(text_feature.shape[0]):
            text = text_feature[i]
            sim = text @ image_embeds.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct = correct + 1
            total = total + 1
        print(total)
        print(correct)
        print(correct/total)
        
        print("image to text")
        i = 0
        correct = 0
        total = 0
        for i in range(image_embeds.shape[0]):
            img = image_embeds[i]
            sim = img @ text_feature.T
            sim = sim.squeeze()
            correct_i = torch.argmax(sim)

            if i==correct_i:
                correct = correct + 1
            total = total + 1
        print(total)
        print(correct)
        print(correct/total)

