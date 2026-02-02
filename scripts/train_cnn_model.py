import pandas as pd
import numpy as np
import glob
import os
import random
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import argparse

from sklearn.metrics import classification_report

import torch
from torch.utils.data import random_split, Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from proteogram.utils import read_yaml


matplotlib.use('agg')
# # For reproducibility, set seeds
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

class ProteogramDataset(Dataset):
    """SCOPe-based Proteograms dataset."""

    def __init__(self, tsv_file, root_dir, pad=True, level='class', new_size=128, transform=None):
        """
        Arguments
        ---------
        tsv_file : string
            Path to the tsv file with annotations.
        root_dir : string
            Directory with all the images.
        level : string
            The scope category level [class, fold, superfamily, family]
        transform : callable, optional
            Optional transform to be applied on a sample.
        """
        self.annot_frame = pd.read_csv(tsv_file, sep='\t')
        self.root_dir = root_dir
        self.files = glob.glob(os.path.join(self.root_dir, '*.jpg'))
        self.transform = transform
        self.pad = pad
        self.level = level # class, fold, superfamily or family
        self.new_size = new_size
        
        # Set label lookups and lists
        self.label_names = []
        for file in self.files:
            # Look up class in annotation dataframe
            bname = os.path.basename(file).replace('.jpg','')
            annot_idx = self.annot_frame[self.annot_frame['PDBFileName'] == bname].index
            if self.level == 'class':
                label = self.annot_frame.loc[annot_idx, 'SCOPeClass'].to_string(index=False)
                self.label_names.append(label)
            elif self.level == 'fold':
                label = self.annot_frame.loc[annot_idx, 'SCOPeFold'].to_string(index=False)
                self.label_names.append(label)
            elif self.level == 'superfamily':
                label = self.annot_frame.loc[annot_idx, 'SCOPeSuperfamily'].to_string(index=False)
                self.label_names.append(label)
            else: # family
                label = self.annot_frame.loc[annot_idx, 'SCOPeFamily'].to_string(index=False)
                self.label_names.append(label)
        self.label_names_unique = set(self.label_names)
        self.labels_to_names = {}
        self.names_to_labels = {}
        for i, name in enumerate(self.label_names_unique):
            self.labels_to_names[i] = name
            self.names_to_labels[name] = i
        self.labels = [self.names_to_labels[n] for n in self.label_names]

    def get_pad(self, curr_size: int, target_size: int):
        d = target_size - curr_size
        if d <= 0: return (0, 0) # no need to pad
        p1 = d // 2
        p2 = d - p1
        return (p1, p2)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = self.files[idx]
        label = torch.tensor(self.labels[idx])
        if self.pad:
            image = plt.imread(img_name)
            H, W = image.shape[0], image.shape[1]
            DH, DW = self.new_size, self.new_size # desired height / width
            padding = (self.get_pad(H, DH), self.get_pad(W, DW), (0, 0))
            image = np.pad(image, padding, constant_values=128) # pad with gray
            image = image[0:DH, 0:DW, :] # crop if needed
        else:
            image = Image.open(img_name).convert('RGB')
            image = image.resize((self.new_size, self.new_size))
            image = np.array(image)
        if self.transform:
            image = self.transform(image)
        return image, label


class ConvNet(nn.Module):
    def __init__(self, num_classes, size):
        super().__init__()

        if size % 8 != 0:
            raise ValueError("Specified input size must be divisible by 8 for this architecture.")
        
        # Calculate final pixel size after 3 pooling layers
        final_pix_size = int(size/2/2/2)

        # Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # The output size after the conv/pool layers needs to be calculated.
        # For example, with 3 conv layers and 3 pooling layers (one after each conv), 
        # a 32x32 image becomes 4x4 pixels with 32 channels. 
        # Total input features for the linear layer: 32 * 4 * 4 = 512
        self.fc_input_features = 32 * final_pix_size * final_pix_size
        # Fully Connected Linear Layers
        self.fc1 = nn.Linear(self.fc_input_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, num_classes) # Output layer, e.g., for num_classes classes 

    def forward(self, x):
        # Conv layer 1, ReLU, Pool
        x = self.pool(F.relu(self.conv1(x)))
        # Conv layer 2, ReLU, Pool
        x = self.pool(F.relu(self.conv2(x)))
        # Conv layer 3, ReLU, Pool
        x = self.pool(F.relu(self.conv3(x)))

        # Dropout to help prevent overfitting
        x = F.dropout(x, training=self.training)

        # Flatten the output from the convolutional layers to a 1D vector
        x = x.view(-1, self.fc_input_features) # -1 infers batch size automatically
        
        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # Output layer (no activation here if using CrossEntropyLoss)
        return x


def train_model(model, train_loader, optimizer, epochs, device=torch.device('cpu')):
    """Train the ConvNet for a certain number of epochs."""
    # Set the model to training mode
    model.train()
    model.to(device)
    training_loss = []

    for epoch in range(epochs):
        train_loss = 0
        # Process the images in batches
        for batch_idx, (data, target) in enumerate(train_loader):
            # Use the CPU or GPU as appropriate
            data, target = data.to(device), target.to(device)
            
            # Reset the optimizer
            optimizer.zero_grad()
            
            # Push the data forward through the model layers
            output = model(data)
            
            # Get the loss
            loss = loss_criteria(output, target)
            
            # Keep a running total
            train_loss += loss.item()
            
            # Backpropagate
            loss.backward()
            optimizer.step()
            
            # # Print metrics for every 10 batches so we see some progress
            # if batch_idx % 10 == 0:
            #     print('Training set [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
            #         batch_idx * len(data), len(train_loader.dataset),
            #         100. * batch_idx / len(train_loader), loss.item()))
                
        # Average loss for the epoch
        avg_loss = train_loss / (batch_idx+1)
        training_loss.append(avg_loss)
        print(f'Epoch {epoch}: Average loss: {avg_loss:.6f}')
    return model

def split_train_test(full_dataset, generator):
    """Split a PyTorch Dataset object into train and test sets"""
    total_size = len(full_dataset)
    train_size = int(total_size * 0.7)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=generator
    )
    return train_dataset, test_dataset

def split_train_val_test(full_dataset, generator):
    """Split a PyTorch Dataset object into train, val, test sets"""
    total_size = len(full_dataset)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.1)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    return train_dataset, val_dataset, test_dataset

def get_accuracies(model, test_loader, class_names, labels_to_names, device=torch.device('cpu')):
    """Accuracies per class"""
    correct_pred = {classname: 0 for classname in class_names}
    total_pred = {classname: 0 for classname in class_names}
    total_correct = 0
    len_data = len(test_loader)
    y_pred = []
    y_test = []
    model.eval()
    model.to(device)
    with torch.no_grad():
        for (data, targets) in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(targets, predictions):
                if label == prediction:
                    total_correct += 1
                    correct_pred[labels_to_names[int(label)]] += 1
                total_pred[labels_to_names[int(label)]] += 1
                y_pred.append(labels_to_names[int(prediction)])
                y_test.append(labels_to_names[int(label)])

    # Print accuracy for each class
    for classname, correct_count in correct_pred.items():
        try:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
        except ZeroDivisionError:
            print(f'No samples in test set for class: {classname}')
    print(f'Overall accuracy: {(total_correct/len_data*100):.1f} %')

    print('\nAdditional Classification Report:')
    print(classification_report(y_test, y_pred))


def view_pred_set(model, test_loader, num_preds, labels_to_names, fig_path):
    """Graph a set of predictions with labels and save plot."""
    predictions = []
    images = []
    labels = []
    with torch.no_grad():
        cnt = 1
        for (data, target) in test_loader:
            images.append(data)
            labels.append(target[0])
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted[0])

            if cnt == num_preds:
                break
            cnt += 1

    for i in range(num_preds):
        plot_row = max(1, int(num_preds/2))
        plt.subplot(2, plot_row, i + 1)
        img = images[i]
        npimg = img.numpy()
        npimg = np.squeeze(npimg, axis=0)
        npimg = npimg / 2 + 0.5
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.axis('off')
        
        color = "green"
        pred = int(predictions[i].numpy())
        label = int(labels[i].numpy())
        name = labels_to_names[label]
        if label != pred:
            color = "red"
        plt.title(name, color=color)

    plt.suptitle('Objects Found by Model', size=20)
    plt.savefig(fig_path)

def load_model(model_path, classes, image_size):
    """Load the ConvNet model from disk."""
    model = ConvNet(classes, image_size)
    ConvNet.load_state_dict(torch.load(model_path))
    return model

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Train CNN on Proteograms.")
    parser.add_argument("--epochs", "-e",
                        type=int,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", "-b",
                        type=int,
                        help="Training batch size.")
    parser.add_argument("--lr", "-l",
                        type=float,
                        help="Training learning rate.")
    parser.add_argument('--overwrite', '-o',
                        action='store_true',
                        help="Recreate / overwrite model")    
    parser.add_argument('--pad_images', '-p',
                        action='store_true',
                        help="Pad images to square dimensions with gray.")
    parser.add_argument('--verbose', '-v',
                        action='store_true',
                        help="Verbose output and logging.")
    args = parser.parse_args()

    config = read_yaml('config.yml')
    root_dir = config['all_proteograms_dir']

    if args.epochs:
        epochs = args.epochs
    elif 'num_epochs_cnn' in config:
        epochs = config['num_epochs_cnn']
    else:
        raise ValueError("Number of epochs must be specified via command line or config.yml")
    if args.lr:
        lr = args.lr
    elif 'learning_rate_cnn' in config:
        lr = config['learning_rate_cnn']
    else:
        raise ValueError("Learning rate must be specified via command line or config.yml")
    if args.batch_size:
        batch_size = args.batch_size
    elif 'batch_size_cnn' in config:
        batch_size = config['batch_size_cnn']
    else:
        raise ValueError("Batch size must be specified via command line or config.yml")
    
    model_file = config.get('cnn_model_file_prefix', 'cnn_proteogram_model') \
        + f'_lr{lr}_bs{batch_size}_e{epochs}.pt'
    model_path = os.path.join(root_dir, model_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_resize = 200

    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    proteogram_dataset = ProteogramDataset(
         tsv_file=os.path.join(root_dir, 'ProteogramData_SCOP_RCSB_PDBe_AnnotationsLookup.tsv'),
         root_dir=root_dir,
         level='class',
         new_size=image_resize,
         pad=args.pad_images,
         transform=transform)
    
    # This is for reproducibility - https://docs.pytorch.org/docs/stable/notes/randomness.html
    g = torch.Generator()
    g.manual_seed(0)

    # Split whole dataset into train and test sets
    train_dataset, test_dataset = split_train_test(
        proteogram_dataset, g)

    class_names = proteogram_dataset.label_names_unique
    # Define torch data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              worker_init_fn=seed_worker,
                                              generator=g)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              worker_init_fn=seed_worker,
                                              generator=g)
    model = ConvNet(len(class_names),
                    image_resize)

    loss_criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model = train_model(model,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        epochs=args.epochs,
                        device=device)

    # Save model
    if os.path.exists(model_path) and not args.overwrite:
        print(f'Model file {model_path} exists and overwrite not set, not saving model.')
    else:
        # Save only the model weights
        torch.save(model.state_dict(), model_path)
        print(f'Saved model to {model_path}')

    get_accuracies(model,
                   test_loader,
                   class_names,
                   proteogram_dataset.labels_to_names)
    
    view_pred_set(model,
                  test_loader,
                  num_preds=10,
                  labels_to_names=proteogram_dataset.labels_to_names,
                  fig_path=os.path.join(root_dir, 'sample_preds.png'))



