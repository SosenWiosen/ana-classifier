from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


url = "https://unsplash.com/photos/8Nc_oQsc2qQ/download?ixid=MnwxMjA3fDB8MXxhbGx8fHx8fHx8fHwxNjcxMjAwNzI0&force=true&w=640"
imagepath='/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff-grouped-filtered/AC2/2D 37_Sample167_AC2_DFS_2D_Image 11.tiff'
image = Image.open(imagepath)

prompts = ["cutlery", "pancakes", "blueberries", "orange juice"]

import torch

inputs = processor(text=prompts, images=[image] * len(prompts), return_tensors="pt")
# predict
with torch.no_grad():
  outputs = model(**inputs)
preds = outputs.logits.unsqueeze(1)

import matplotlib.pyplot as plt

_, ax = plt.subplots(1, len(prompts) + 1, figsize=(3*(len(prompts) + 1), 4))
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(len(prompts))];
[ax[i+1].text(0, -15, prompt) for i, prompt in enumerate(prompts)];