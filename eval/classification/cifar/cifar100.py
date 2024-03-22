import torch
import sys
sys.path.append('../../..')
from model import longclip
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from templates import imagenet_templates

from tqdm import tqdm


def zeroshot_classifier(model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates]  # format with class
            texts = longclip.tokenize(texts).cuda()  # tokenize
            class_embeddings = model.encode_text(texts)  # embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("../../../checkpoints/longclip-B.pt", device=device)
model.eval()

testset = torchvision.datasets.CIFAR100(root="data/cifar100", train=False, download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=8)

text_feature = zeroshot_classifier(model, testset.classes, imagenet_templates)

correct = 0
total = 0
with torch.no_grad():
    i = 0
    for data in testset:
        images, labels = data
        images = preprocess(images).unsqueeze(0).to(device)

        image_feature = model.encode_image(images)
        image_feature = image_feature/image_feature.norm(dim=-1, keepdim=True)
        
        sims = image_feature @ text_feature
        
        pred = torch.argmax(sims, dim=1).item()
        if labels == pred:
            correct += 1
        total += 1

    print(correct)
    print(total)
    print(correct/total)

print("Accuracy of the CLIP model on the CIFAR-10 test images: %d %%" % (100 * correct / total))