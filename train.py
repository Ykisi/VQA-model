import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from Dataset import VQA_Dataset
from Model import VQAModel
from time import time
import matplotlib.pyplot as plt


def train(cache_path):

    train_dataset = VQA_Dataset(cache_path, 'train')
    train_loader = data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_dataset = VQA_Dataset(cache_path, 'val')
    val_loader = data.DataLoader(val_dataset, batch_size=400, shuffle=True)
    vqa_model = VQAModel(num_q_tokens=len(train_dataset.q_dict), ans_vocab_size=len(train_dataset.ans2label),
                         hidden_size=512, hidden_dim=1024, embedding_dim=300)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vqa_model = vqa_model.to(device)
    critarion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(vqa_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    val_scores = []
    train_scores = []
    val_loss = []
    train_loss = []

    for epoch in range(20):
        epoch_time = time()

        epoch_loss = 0.0
        v_loss = 0.0
        val_score = 0.0
        train_score = 0.0

        vqa_model.train()
        for batch, (image, question, annotation) in enumerate(train_loader):
            image = image.to(device)
            question = question.to(device)
            annotation = annotation.to(device)
            output = vqa_model(image, question)
            loss = critarion(output, annotation.argmax(dim=1))
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(vqa_model.parameters(), 0.25)
            optimizer.step()
            epoch_loss += loss.item() * question.size(0)
            result_score = nn.functional.one_hot(output.argmax(dim=1), num_classes=len(val_dataset.ans2label))
            train_score += torch.sum(result_score * annotation).item()

        epoch_loss /= len(train_dataset)

        train_loss.append(epoch_loss)

        train_score /= len(train_dataset)
        train_score *= 100
        train_scores.append(train_score)

        val_score = 0.0
        vqa_model.eval()
        with torch.no_grad():
            for batch, (image, question, annotation) in enumerate(val_loader):
                image = image.to(device)
                question = question.to(device)
                annotation = annotation.to(device)
                output = vqa_model(image, question)
                result_score = nn.functional.one_hot(output.argmax(dim=1), num_classes=len(val_dataset.ans2label))
                val_score += torch.sum(result_score * annotation).item()
                loss = critarion(output, annotation.argmax(dim=1))
                v_loss += loss.item() * question.size(0)

        v_loss /= len(val_dataset)
        val_loss.append(v_loss)

        val_score /= len(val_dataset)
        val_score *= 100
        val_scores.append(val_score)

        scheduler.step()

        print(f'Epoch {epoch} Train Loss {epoch_loss:.4f}'
              f' Validation Score {val_score:.3f} done in {time() - epoch_time:.2f}s')

    val_error = [(1 - (x / 100)) for x in val_scores]
    train_error = [(1 - (x / 100)) for x in train_scores]

    # plot loss:
    epochs = list(range(1, 21))
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.ylim((0, 3.5))
    plt.show()

    # plot error:
    plt.plot(epochs, train_error, 'g', label='Training Error')
    plt.plot(epochs, val_error, 'b', label='Validation Error')
    plt.title('Training and Validation Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.ylim((0, 1))
    plt.show()
