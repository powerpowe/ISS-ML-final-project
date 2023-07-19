import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from sklearn import metrics
import PIL

def training(model, train_dset, valid_dset, device, verbose, hyperparameter, curve_name):
    """
    hyperparameter := {'batch_size': ...,
                       'lr': ...,
                        'epoch': ...}

    """
    train_dloader = DataLoader(train_dset, hyperparameter['batch_size'], True, drop_last=True)
    valid_dloader = DataLoader(valid_dset, hyperparameter['batch_size'], True, drop_last=False)

    train_accuracy_log, train_loss_log, valid_accuracy_log, valid_loss_log = [], [], [], []

    model.to(device)
    loss_function = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), hyperparameter['lr'])

    epochs = hyperparameter['epoch']

    for epoch in range(epochs):
        if verbose: print(f"Epoch: {epoch + 1}")
        correct = 0

        # training
        for x_train, y_train in train_dloader:
            model.train()
            x_train = x_train.to(device)
            y_train = y_train.to(device)
            criterion = model(x_train)

            y_train_onehot = torch.zeros(len(y_train), 2).to(device)
            y_train_onehot[range(len(y_train)), y_train] = 1

            loss = loss_function(criterion, y_train_onehot)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model_suggestion = torch.argmax(criterion, 1)
            correct += torch.sum(model_suggestion == y_train).item()

        # accuracy, loss
        train_accuracy_log.append(correct / len(train_dset))
        train_loss_log.append(loss.item())

        if verbose:print(f"train set Acc: {correct / len(train_dset)}, loss: {loss.item()}")

        # validation
        with torch.no_grad():
            model.eval()
            correct = 0

            for x_valid, y_valid in valid_dloader:
                x_valid = x_valid.to(device)
                y_valid = y_valid.to(device)
                criterion = model(x_valid)

                y_valid_onehot = torch.zeros(len(y_valid), 2).to(device)
                y_valid_onehot[range(len(y_valid)), y_valid] = 1
                loss = loss_function(criterion, y_valid)

                model_suggestion = torch.argmax(criterion, 1)
                correct += torch.sum(model_suggestion == y_valid).item()

            # accuracy, loss
            valid_accuracy_log.append(correct / len(valid_dset))
            valid_loss_log.append(loss.item())

            if verbose: print(f"valid set Acc: {correct / len(valid_dset)}, loss: {loss.item()}")

    # plotting
    fig = plt.figure(figsize=(8, 8))
    plt.title('Accuray Curve')
    plt.plot(train_accuracy_log, label='train_accuracy')
    plt.plot(valid_accuracy_log, label='valid_accuracy')
    plt.legend()
    plt.savefig(f'./curve/accuracy_curve_{curve_name}.png')


    fig = plt.figure(figsize=(8, 8))
    plt.plot(train_loss_log, label='train_loss')
    plt.plot(valid_loss_log, label='valid_loss')
    plt.legend()
    plt.savefig(f'./curve/loss_curve_{curve_name}.png')

    if verbose: print('curve saved')

    return model

def testing(model, test_dset, device):

    test_dloader = DataLoader(test_dset, 16, False, drop_last=False)
    whole_model_suggestion = []
    whole_answer = []
    with torch.no_grad():
        model.eval()

        for x_test, y_test in test_dloader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)
            criterion = model(x_test)

            model_suggestion = torch.argmax(criterion, 1)
            whole_model_suggestion += list(model_suggestion.to('cpu'))
            whole_answer += list(y_test.to('cpu'))

        print(f"Test Accuracy: {metrics.accuracy_score(whole_answer, whole_model_suggestion)}")
        print(f"Test F1 score: {metrics.f1_score(whole_answer, whole_model_suggestion)}")
        print(f"Test Precision: {metrics.precision_score(whole_answer, whole_model_suggestion)}")
        print(f"Test Recall: {metrics.recall_score(whole_answer, whole_model_suggestion)}")

def inference_myface(model, device):
    face = PIL.Image.open('./data/myface/myface.jpg')
    temp = TF.to_tensor(face)
    face_in = TF.resize(temp, (224, 224))
    with torch.no_grad():
        model.eval()
        face_in = face_in.unsqueeze(0).to(device)
        print(nn.functional.softmax(model(face_in).to('cpu'), 1))