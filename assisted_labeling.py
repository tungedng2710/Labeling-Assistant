"""
My GitHub: https://github.com/tungedng2710
Labeling Assistant using OpenClip foundation models
For instant usage, just install
    $ pip install open_clip_torch
For advanced purposes, checkout the OpenClip official repository at:
    https://github.com/mlfoundations/open_clip
"""
import os
import numpy as np
import time
from datetime import timedelta
from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import open_clip


class OpenClipDataset(Dataset):
    """
    Pytorch Dataset with OpenClip preprocessing 
    """
    def __init__(self, root_dir, preprocess, classes, device):
        self.root_dir = root_dir
        self.preprocess = preprocess
        self.classes = classes
        self.device = device
        image_names= os.listdir(root_dir)
        self.image_paths = [os.path.join(root_dir, name) \
                            for name in image_names if check_img_file(name)]

    def __getitem__(self, idx):
        assert idx < len(self.image_paths)
        image_path = self.image_paths[idx]
        return image_path, self.preprocess(Image.open(image_path)).to(self.device)

    def __len__(self):
        return len(self.image_paths)


def check_img_file(filename):
    """
    Consider whether it's image file
    """
    return filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))


def predict(model, inputs, classes, tokenizer, device):
    """
    Run model on input images
    Args:
        - model: open_clip pretrained model
        - inputs: 4-dim pytorch tensor: (bs, d, w, h)
        - classes: list of label name to classify
        - tokenizer: open_clip text tokenizer
        - device: pytorch device
    """
    model.to(device)
    text = tokenizer(classes).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(inputs)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    _, labels = torch.max(text_probs, dim=1)
    return labels


if __name__ == '__main__':
    # Config
    root_dir = "./test_samples"
    classes = ["baby", "children", "teenager", "adult", "senior"]

    pretrained = ('ViT-L-14', 'commonpool_xl_s13b_b90k') # open_clip.list_pretrained()
    model, _, preprocess = open_clip.create_model_and_transforms(*pretrained)
    tokenizer = open_clip.get_tokenizer(pretrained[0])

    device = torch.device("cuda:0")
    batch_size = 1
    save = True # save results to file

    # Inference
    start = time.time()
    predictions = []
    dataset = OpenClipDataset(root_dir, preprocess, classes, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for paths, features in tqdm(dataloader, desc ="Inference"):
        results = predict(model, features, dataset.classes, tokenizer, device)
        for idx in range(len(paths)):
            prediction = f"{paths[idx]} \t {results[idx].item()}"
            predictions.append(prediction)
    if save:
        with open('results.txt', 'w') as f:
            for prediction in predictions:
                f.write(f"{prediction}\n")

    end = time.time()
    print(f"Running time: {timedelta(seconds=round(end - start, 0))}")