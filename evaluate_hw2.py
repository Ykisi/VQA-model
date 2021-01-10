import torch
import torch.nn as nn
import torch.utils.data as data
from Dataset import VQA_Dataset
from pathlib import Path
from data_preprocess import question_dict, image_process

@torch.no_grad()
def evaluate_hw2():
    data_path = Path('/datashare/')
    cache_path = Path('./')
    data_files = ['dict.pkl', 'val_img.hdf5', 'val_img_dict.pkl']
    if not all((cache_path / file).is_file() for file in data_files):
        question_dict(cache_path)
        image_process(data_path, cache_path, 'val')
    val_dataset = VQA_Dataset(cache_path, 'val')
    val_loader = data.DataLoader(val_dataset, batch_size=200, shuffle=True)
    vqa_model = torch.load('vqa_model20.pkl')
    vqa_model.eval()
    score = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for batch, (image, question, annotation) in enumerate(val_loader):
        image = image.to(device)
        question = question.to(device)
        annotation = annotation.to(device)
        output = vqa_model(image, question)
        result_score = nn.functional.one_hot(output.argmax(dim=1), num_classes=len(val_dataset.ans2label))
        score += torch.sum(result_score * annotation).item()

    score /= len(val_dataset)
    score *= 100

    return score


if __name__ == '__main__':
    print(evaluate_hw2())