import sys
sys.path.append('../..')
from model import longclip
import torch
from torchvision.datasets import CocoCaptions
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("../../checkpoints/longclip-B.pt", device=device)

model.eval()

coco = CocoCaptions(root="data/coco/val2017/", annFile="data/coco/annotations/captions_val2017.json", transform=None)

image_features = []
text_features = []
pred_true = 0

with torch.no_grad():
    for image, captions in coco:
        image_input = preprocess(image).unsqueeze(0).to(device)
        image_features.append(model.encode_image(image_input))

        captions = captions[0:5]
        caption_input = longclip.tokenize(captions).to(device)
        text_features.extend(model.encode_text(caption_input))

    image_features = torch.stack(image_features).squeeze()
    image_features /= image_features.norm(dim=-1, keepdim=True)

    print(image_features.shape)
    text_features = torch.stack(text_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    similarity = image_features.squeeze() @ text_features.squeeze().T
   
    print("I2T")
    for i in range(5000):
        pred = similarity[i]
        b = pred.argsort()[-1:]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in b:
                pred_true = pred_true + 1
                break
    print(pred_true / 5000)
    pred_true = 0

    for i in range(5000):
        pred = similarity[i]
        b = pred.argsort()[-5:]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in b:
                pred_true = pred_true + 1
                break
    print(pred_true / 5000)
    pred_true = 0

    for i in range(5000):
        pred = similarity[i]
        b = pred.argsort()[-10:]
        for j in range(5):
            true_index = 5 * i + j
            if true_index in b:
                pred_true = pred_true + 1
                break
    print(pred_true / 5000)
    pred_true = 0

    print("T2I")
    similarity = similarity.T
    for i in range(25000):
        pred = similarity[i]
        b = pred.argsort()[-1:]
        true_index = i//5
        if true_index in b:
            pred_true = pred_true + 1

    print(pred_true/25000)
    pred_true = 0

    for i in range(25000):
        pred = similarity[i]
        b = pred.argsort()[-5:]
        true_index = i//5
        if true_index in b:
            pred_true = pred_true + 1

    print(pred_true/25000)
    pred_true = 0

    for i in range(25000):
        pred = similarity[i]
        b = pred.argsort()[-10:]
        true_index = i//5
        if true_index in b:
            pred_true = pred_true + 1

    print(pred_true/25000)
    