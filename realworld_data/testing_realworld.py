import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn import metrics
import os
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import transforms
import pandas as pd


class RealWorldDataSet(Dataset):

    def __init__(self):
        self.path = './realworld_data'

        label_data = pd.read_csv('./realworld_data/your_prediction.csv', header=0)
        self.video_name = label_data[['video']].values
        self.face_name = label_data[['face']].values
        self.label = label_data[['label']].values

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return 76


    def __getitem__(self, idx):
        img = read_image(self.path + '/' + self.video_name[idx][0] + '/' + self.face_name[idx][0])
        img = img / 255  # [0, 1]

        label = self.label[idx][0]
        return self.transform(img), label


def testing(model, test_dset, device):
    test_dloader = DataLoader(test_dset, 16, shuffle=False, drop_last=False)
    whole_model_suggestion = []
    whole_answer = []
    with torch.no_grad():
        model.eval()

        for x_test, y_test in test_dloader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            criterion = model(x_test)

            model_suggestion = torch.argmax(criterion, 1)
            print(model_suggestion)
            whole_model_suggestion += list(model_suggestion.to('cpu'))
            whole_answer += list(y_test.to('cpu'))

        print(f"Test Accuracy: {metrics.accuracy_score(whole_answer, whole_model_suggestion)}")
        print(f"Test F1 score: {metrics.f1_score(whole_answer, whole_model_suggestion)}")
        print(f"Test Precision: {metrics.precision_score(whole_answer, whole_model_suggestion)}")
        print(f"Test Recall: {metrics.recall_score(whole_answer, whole_model_suggestion)}")

def testing_ensemble(model1, model2, test_dset, device):
    test_dloader = DataLoader(test_dset, 16, shuffle=False, drop_last=False)
    whole_model_suggestion = []
    whole_answer = []
    with torch.no_grad():
        model1.eval()
        model2.eval()

        for x_test, y_test in test_dloader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            criterion = (model1(x_test) + model2(x_test)) / 2

            model_suggestion = torch.argmax(criterion, 1)
            print(model_suggestion)
            whole_model_suggestion += list(model_suggestion.to('cpu'))
            whole_answer += list(y_test.to('cpu'))

        print(f"Test Accuracy: {metrics.accuracy_score(whole_answer, whole_model_suggestion)}")
        print(f"Test F1 score: {metrics.f1_score(whole_answer, whole_model_suggestion)}")
        print(f"Test Precision: {metrics.precision_score(whole_answer, whole_model_suggestion)}")
        print(f"Test Recall: {metrics.recall_score(whole_answer, whole_model_suggestion)}")