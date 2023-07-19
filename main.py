import torch.cuda
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from module import utils, training_testing

IMG_TRANSFORM = T.Compose([T.ToTensor(), T.Resize((224, 224))])

if __name__ == "__main__":
    for d in ['cats_and_dogs_small', 'Real_Fake_Dataset']:
        for e in [1e-2, 1e-3, 1e-4]:
            print(d, e, 'setting')
            setting = {"model_name": "xception",  # xception
                       'dataset': d}  # cats_and_dogs_small or Real_Fake_Dataset

            hyperparameter = {'batch_size': 32,
                              'lr': e,
                              'epoch': 10}
            model = utils.download_model(setting["model_name"], 2, True)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            train_dset = ImageFolder(root=f"./data/{setting['dataset']}/train",
                                     transform=IMG_TRANSFORM)
            valid_dset = ImageFolder(root=f"./data/{setting['dataset']}/validation",
                                     transform=IMG_TRANSFORM)

            model = training_testing.training(model, train_dset, valid_dset, device, False, hyperparameter, f"{d}-{e}")

            model.to(device)

            test_dset = ImageFolder(root=f"./data/{setting['dataset']}/test",
                                     transform=IMG_TRANSFORM)

            training_testing.testing(model, test_dset, device)

            training_testing.inference_myface(model, device)