import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm


class V_encoder(nn.Module):  # VGG
    def __init__(self, hidden_dim):
        super(V_encoder, self).__init__()
        self.in_channels = 3
        layers = []
        for size in [64, 2, 128, 2, 256, 256, 2, 512, 512, 2, 512, 512, 2]:
            if size == 2:
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(self.in_channels, size, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(size), nn.ReLU(inplace=True)]
                self.in_channels = size
        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        fc = [
            weight_norm(nn.Linear(512 * 7 * 7, 4096), dim=None),
            nn.ReLU(),
            nn.Dropout(0.5),
            weight_norm(nn.Linear(4096, 4096), dim=None),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, hidden_dim)
        ]
        self.classifier = nn.Sequential(*fc)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        l2_norm = nn.functional.normalize(x, p=2, dim=1).detach()
        return l2_norm


class Q_encoder(nn.Module):
    def __init__(self, num_q_tokens, embedding_dim, hidden_size, hidden_dim, drop=0.56):
        super(Q_encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_q_tokens, embedding_dim=embedding_dim, padding_idx=0)
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(drop)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=3, dropout=0.56)
        self.fc = nn.Linear(2 * 3 * hidden_size, hidden_dim)

    def forward(self, question):
        qu_embedding = self.embedding(question)
        qu_embedding = self.tanh(qu_embedding)
        qu_embedding = qu_embedding.transpose(0, 1)
        _, (hidden, cell) = self.lstm(qu_embedding)
        qu_feature = torch.cat((hidden, cell), dim=2)
        qu_feature = qu_feature.transpose(0, 1)
        qu_feature = qu_feature.reshape(qu_feature.size()[0], -1)
        qu_feature = self.tanh(qu_feature)
        qu_feature = self.fc(qu_feature)
        return qu_feature


class VQAModel(nn.Module):

    def __init__(self, num_q_tokens, ans_vocab_size, hidden_size, hidden_dim, embedding_dim):
        super(VQAModel, self).__init__()
        self.V_encoder = V_encoder(hidden_dim)
        self.Q_encoder = Q_encoder(num_q_tokens, embedding_dim, hidden_size, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(hidden_dim, ans_vocab_size)
        self.fc2 = nn.Linear(ans_vocab_size, ans_vocab_size)

    def forward(self, image, question):
        v_feature = self.V_encoder(image)
        q_feature = self.Q_encoder(question)
        x = torch.mul(v_feature, q_feature)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x
