"""
Finetune model based on transfer learning in the following tutorial:
https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
Watch out for emtpy folders in the DataLoader (remove any empty folders).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torch.utils.data import WeightedRandomSampler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.transforms import v2
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

from proteogram.utils import read_yaml


config = read_yaml('config.yml')

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': v2.Compose([
        v2.RandomAutocontrast(p=0.1),
        v2.RandomApply(
            [v2.ColorJitter(brightness=0,hue=0,saturation=0.3)],
            p=0.1),
        v2.RandomApply(
            [v2.RandomAdjustSharpness(sharpness_factor=2)],
            p=0.1),
        v2.RandomApply(
            [v2.RandomErasing(p=0.5)],
            p=0.1),
        v2.Resize(224),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': v2.Compose([
        v2.Resize(224),
        v2.ToTensor(),
        v2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = config['training_data_dir']
model_file = config['model_file']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create datasets
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# Get sample weights from dataset indices and labels/targets
y_train = [image_datasets['train'].targets[i] for i in range(len(image_datasets['train'].samples))]
class_sample_count = np.array(
    [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
weight = 1. / class_sample_count
sample_weights = np.array([weight[t] for t in y_train])
sample_weights = torch.from_numpy(sample_weights)

# Sampler to deal with class imbalance
sampler = WeightedRandomSampler(sample_weights.type('torch.DoubleTensor'), len(sample_weights))

# Create data loaders
dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'],
                                                   batch_size=config['batch_size'],
                                                   num_workers=8,
                                                   shuffle=True)
dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'],
                                                 batch_size=config['batch_size'],
                                                 num_workers=8,
                                                 shuffle=True)
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

# Train model function
def train_model(model, criterion, optimizer, scheduler, num_epochs=10):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                if phase == 'val':
                    scheduler.step(epoch_loss)

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path, weights_only=True))
    return model

# Load pre-trained model
if config['pretrained'] == 'True':
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
else:
    # No pretrained weights (no transfer learning)
    model_ft = models.resnet18()
num_ftrs = model_ft.fc.in_features
print(num_ftrs)
## Here the size of each output sample is set to 2.
## Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
model_ft.fc = nn.Linear(num_ftrs, len(class_names))

# VGG19
#model_ft = models.vgg19(weights='IMAGENET1K_V1')
#num_ftrs = model_ft.classifier[0].in_features

#input_size = model_ft.classifier[0].in_features    
#model_ft.classifier = nn.Sequential(
#    nn.Linear(input_size, 128), nn.ReLU(),
#    nn.Linear(128, len(class_names)))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(),
                         lr=config['learning_rate'],
                         momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
#exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', factor=0.1)

# Train and evaluate (all done in train_model function)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=config['num_epochs'])

# Save model as file
try:
    torch.save(model_ft, os.path.join(data_dir, f"resnet18_lr{config['learning_rate']}_bs{config['batch_size']}.pt"))
except:
    torch.save(model_ft, os.path.join(data_dir, 'resnet18_new.pt'))

