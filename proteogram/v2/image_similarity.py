"""
Image search approach based on https://github.com/totogot/ImageSimilarity
"""
import torch
import os
import math
import json
from time import time
import glob
from torch.multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import vit_b_16
import torchvision.transforms as transforms

from kmeans_pytorch import kmeans
from PIL import Image, ImageDraw, ImageFont


class Img2Vec:
    """
    Class for embedding dataset of image files into vectors using Pytorch
    standard neural networks.

    Parameters:
    -----------
    model_name: str object specifying neural network architecture to utilise.
        Must align to the naming convention specified on Pytorch documentation:
        https://pytorch.org/vision/main/models.html#classification
        For supported model architectures see self.embed_dict below
    weights: str object specifying the pretrained weights to load into model or
        a file path to a supported ResNet model.
        Only weights supported by Pytorch torchvision library can be accessed.
        Current functionality reverts to DEFAULT weights if no specified.

    See also:
    -----------
    Img2Vec.embed_dataset(): embed passed images as feature vectors
    Img2Vec.save_dataset(): save embedded dataset to file for future loading
    Img2Vec.load_dataset(): load previously embedded dataset of feature vectors
    Img2Vec.similarities(): calculate cosine similarities for the embedding dataset
    Img2Vec.cluster_dataset(): group embedded images into specified n clusters

    Example:
    -----------

    ImgSim = imgsim.Img2Vec('resnet50', weights='DEFAULT')
    ImgSim.embed_dataset('[EXAMPLE PATH TO DIRECTORY OF IMAGES]')

    ImgSim.save_dataset('[OUTPUT PATH FOR SAVING EMBEDDEDINGS]')

    ImgSim.similarities(n=5)

    ImgSim.cluster_dataset(nclusters=6, display=True)
    """

    def __init__(self, model_name_or_path, dataset_dir, embed_file=None, weights="DEFAULT", device=None):
        # dictionary defining the supported NN architectures
        self.embed_dict = {
            "resnet50": self.obtain_children,
            "resnet_ft": self.obtain_children,
            "resnet152": self.obtain_children,
            "resnext50_32x4d": self.obtain_children,
            "resnext101_64x4d": self.obtain_children,
            "convnext_large": self.obtain_children,
            "vgg19": self.obtain_classifier,
            "efficientnet_b0": self.obtain_classifier,
            "vit_b_16": self.obtain_encoder,
        }
        if not device:
            self.device = self.set_device()
        else:
            self.device = torch.device(device)

        # assign class attributes
        self.architecture = self.validate_model(model_name_or_path)
        if self.architecture == "resnet_ft":
            weights = model_name_or_path
        self.vit_model = None
        if self.architecture == "vit_b_16":
            self.vit_model = vit_b_16()
            self.vit_model = self.vit_model.to(self.device).eval()
        self.weights = weights
        self.transform = self.assign_transform(weights)
        self.model = self.initiate_model()
        self.embed = self.assign_layer()
        self.embed_file = embed_file
        self.cosine = nn.CosineSimilarity(dim=1)
        self.dataset = {}
        self.image_clusters = {}
        self.cluster_centers = {}
        self.sim_dict = {}
        self.files = self.validate_source(dataset_dir)

    def validate_model(self, model_name_or_path):
        if os.path.exists(model_name_or_path):
            model_name = "resnet_ft"
        else:
            if model_name_or_path not in self.embed_dict.keys():
                raise ValueError(f"The model {model_name_or_path} is not supported or is not found.")
            else:
                model_name = model_name_or_path
        return model_name

    def assign_transform(self, weights):
        weights_dict = {
            "resnet50": models.ResNet50_Weights,
            "resnet152": models.ResNet152_Weights,
            "resnext50_32x4d": models.ResNeXt50_32X4D_Weights,
            "resnext101_64x4d": models.ResNeXt101_64X4D_Weights,
            "convnext_large": models.ConvNeXt_Large_Weights,
            "vgg19": models.VGG19_Weights,
            "efficientnet_b0": models.EfficientNet_B0_Weights,
            "vit_b_16": models.ViT_B_16_Weights,
            "resnet_ft": None
        }

        # try load preprocess from torchvision else assign default
        try:
            w = weights_dict[self.architecture]
            weights = getattr(w, weights)
            preprocess = weights.transforms()
        except Exception:
            preprocess = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.Lambda(lambda img: img.convert('RGB')),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

        return preprocess

    def set_device(self):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def initiate_model(self):
        if self.architecture == "resnet_ft":
            loaded = torch.load(self.weights, weights_only=False,
                                map_location=self.device)
            if isinstance(loaded, nn.Module):
                model = loaded
            else:
                # State dict saved with torch.save(model.state_dict()).
                # Auto-detect architecture from key signatures.
                model = self._model_from_state_dict(loaded)
                model.load_state_dict(loaded)
        else:
            m = getattr(models, self.architecture)
            model = m(weights=self.weights)
        model.to(self.device)
        return model.eval()

    def _model_from_state_dict(self, state_dict):
        """Reconstruct a model object from a bare state dict.

        Detects architecture by inspecting state dict keys:
          - 'fc.1.weight'  -> ResNet18 with nn.Sequential(Dropout, Linear) head
                              (saved by train_multiple_models.py --model resnet18)
          - 'fc.weight'    -> ResNet18 with plain nn.Linear head (legacy format)
          - 'fc2.weight'   -> From-scratch ConvNet
                              (saved by train_multiple_models.py --model cnn)
        """
        import torchvision.models as tv_models

        if 'fc.1.weight' in state_dict:
            # ResNet18 with nn.Sequential(Dropout(0.5), Linear) head
            num_classes = state_dict['fc.1.weight'].shape[0]
            model = tv_models.resnet18()
            model.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.fc.in_features, num_classes),
            )
        elif 'fc.weight' in state_dict:
            # ResNet18 with plain nn.Linear head (legacy)
            num_classes = state_dict['fc.weight'].shape[0]
            model = tv_models.resnet18()
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif 'fc2.weight' in state_dict:
            # From-scratch ConvNet
            num_classes = state_dict['fc2.weight'].shape[0]
            model = self._build_convnet(num_classes)
        else:
            raise ValueError(
                f"Cannot infer model architecture from state dict keys. "
                f"First 10 keys: {list(state_dict.keys())[:10]}"
            )
        return model

    @staticmethod
    def _build_convnet(num_classes):
        """Reconstruct the from-scratch ConvNet used in train_multiple_models.py."""
        import torch.nn.functional as F

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
            )

        class _ConvNet(nn.Module):
            def __init__(self, n):
                super().__init__()
                self.block1 = _block(3, 64)
                self.block2 = _block(64, 128)
                self.block3 = _block(128, 256)
                self.block4 = _block(256, 256)
                self.gap = nn.AdaptiveAvgPool2d(1)
                self.fc1 = nn.Linear(256, 128)
                self.fc2 = nn.Linear(128, n)

            def forward(self, x):
                x = self.block1(x); x = self.block2(x)
                x = self.block3(x); x = self.block4(x)
                x = self.gap(x).view(x.size(0), -1)
                x = F.dropout(F.relu(self.fc1(x)), p=0.5, training=self.training)
                return self.fc2(x)

        return _ConvNet(num_classes)

    def assign_layer(self):
        model_embed = self.embed_dict[self.architecture]()

        return model_embed

    def obtain_children(self):
        children = list(self.model.children())
        # ConvNet: children[-1]=fc2, children[-2]=fc1 — strip both and add Flatten
        # so the GAP output (batch, 256, 1, 1) is correctly flattened to (batch, 256).
        # ResNet18: children[-1]=fc, children[-2]=avgpool — strip only fc; the
        # (batch, 512, 1, 1) avgpool output works fine with CosineSimilarity(dim=1).
        if isinstance(children[-2], nn.Linear):
            model_embed = nn.Sequential(*children[:-2], nn.Flatten())
        else:
            model_embed = nn.Sequential(*children[:-1])
        return model_embed

    def obtain_classifier(self):
        self.model.classifier = self.model.classifier[:-1]

        return self.model
    
    def obtain_encoder(self):
        model_embed = nn.Sequential(*list(self.model.children())[:-1])

        # The encoder only
        model_embed = model_embed[1]

        return model_embed

    def directory_to_list(self, dir):
        ext = (".png", ".jpg", ".jpeg")

        d = os.listdir(dir)
        source_list = [os.path.join(dir, f) for f in d if \
                       os.path.splitext(f)[1].lower() in ext]

        return source_list

    def validate_source(self, source):
        # convert source format into standard list of file paths
        if isinstance(source, list):
            source_list = [f for f in source if os.path.isfile(f)]
        elif os.path.isdir(source):
            ext = ["png", "jpg", "jpeg"]
            #source_list = self.directory_to_list(source)
            source_list = glob.glob(os.path.join(source, '**', '*.*'),
                                    recursive=True)
            source_list = [f for f in source_list if \
                           f.split('.')[-1].lower() in ext]
        elif os.path.isfile(source):
            source_list = [source]
        else:
            raise ValueError('"source" expected as file, list or directory.')

        return source_list

    def embed_image_for_mp(self, img_file):
        # load and preprocess image
        img = Image.open(img_file)
        with torch.no_grad():
            img_trans = self.transform(img)

            # store computational graph on GPU if available
            if self.device == "cuda:0":
               img_trans = img_trans.cuda()

            img_trans = img_trans.unsqueeze(0)

        return (img_file, self.embed(img_trans))

    def embed_image(self, img_file):
        # load and preprocess image
        img = Image.open(img_file)
        with torch.no_grad():
            if self.architecture == "vit_b_16":
                img_trans = self.transform(img)
                img_trans = img_trans.unsqueeze(0).to(self.device)
                img_trans = self.vit_model._process_input(img_trans)
                n = img_trans.shape[0]
                batch_class_token = self.vit_model.class_token.expand(n, -1, -1)
                img_trans = torch.cat([batch_class_token, img_trans], dim=1)
                embedded_img = self.embed(img_trans)[:, 0]
            else:
                img_trans = self.transform(img)
                img_trans = img_trans.unsqueeze(0)
                img_trans = img_trans.to(self.device)
                embedded_img = self.embed(img_trans)

        return embedded_img

    def embed_dataset_mp(self, source):
        # convert source to appropriate format
        self.files = self.validate_source(source)
        with torch.no_grad():
            with Pool(1) as pool:
                results = pool.map(self.embed_image_for_mp, self.files)
            for img, embedding in results:
                self.dataset[os.path.basename(str(img))] = embedding.clone()#.detach().item()

    def embed_dataset(self):
        # convert source to appropriate format
        with torch.no_grad():
            for img in tqdm(self.files):
                self.dataset[os.path.basename(str(img))] = self.embed_image(img)

    def sim_calc(self, image_path1, image_path2):
        embedding1 = self.dataset[image_path1]
        embedding2 = self.dataset[image_path2]
        with torch.no_grad():
            sim = self.cosine(embedding1, embedding2)[0].item()
        return (sim, image_path1, image_path2)

    def embed_and_sim_calc(self, image_paths):
        image_path1, image_path2 = image_paths
        with torch.no_grad():
            _, embedding1 = self.embed_image(image_path1)
            _, embedding2 = self.embed_image(image_path2)
            sim = self.cosine(embedding1, embedding2)[0].item()
        return (sim, image_path1, image_path2)

    def sim_calc_new_embedding(self,
                               image_path_query,
                               image_path_target,
                               embedding_query,
                               embedding_target):
        with torch.no_grad():
            sim = self.cosine(embedding_query, embedding_target)[0].item()
        return (sim, image_path_target)

    def similarities_new_image(self,
                               query_image_path,
                               n=10,
                               save_results_dir=None,
                               save_result_images_dir=None):
        embedding_new = self.embed_image(query_image_path).detach()

        with mp.Pool(os.cpu_count()-2) as pool:
           results = pool.starmap(self.sim_calc_new_embedding,
                               [(query_image_path, image_path_j, embedding_new, embedding_j) for\
                                 (image_path_j, embedding_j) in self.dataset.items()])
        
        scores = {}
        for (sim, image_path_j) in results:
            scores[image_path_j] = sim
        
        scores_n_arr = sorted(scores.items(),
                            key=lambda item: item[1],
                            reverse=True)[:n]

        # # If there's a dir specified in save_result_images_dir, create result image
        self.save_images(query_image_path,
                         save_result_images_dir,
                         scores_n_arr=scores_n_arr)

        return scores_n_arr

    def sim_calc_new_embedding_scratch(self,
                               image_path_query,
                               image_path_target):
        with torch.no_grad():
            embedding_query = self.embed_image(image_path_query)
            embedding_target = self.embed_image(image_path_target)
            sim = self.cosine(embedding_query, embedding_target)[0].item()
        return (image_path_query, image_path_target, sim)

    def similarities_new_image_scratch(self,
                               image_path_query,
                               n=10,
                               save_result_images_dir=None,
                               use_prev_embeddings=False):
        start_sim_calc = time()
        # Check if need to create embeddings for input dataset
        if use_prev_embeddings == False:
            print(f'Creating and saving embedding dataset')
            self.embed_dataset()
            self.save_dataset()
        else:
            print('Using previously embedded dataset')
            self.load_dataset()

        # initiate computation of consine similarity
        cosine = nn.CosineSimilarity(dim=1).to(self.device)
        # Create a dict of similarities (a dict of dict of scores), e.g:
        # this looks like --> sim_dict[image_i] = {dict[image_0], dict[image_1], ...}
        scores = []
        embedding_query = self.embed_image(image_path_query)
        for image_path_i, embedding_i in tqdm(self.dataset.items()):
            sim = cosine(embedding_query, embedding_i)[0].item()
            scores.append((image_path_i, sim))
        # Sort the scores
        scores = sorted(scores,
                        key=lambda item: item[1],
                        reverse=True)
        scores_n_arr = scores[:n]
        sim_calc_time = time() - start_sim_calc

        # If there's a dir specified in save_result_images_dir, create result image
        if save_result_images_dir:
            self.save_images(image_path_query,
                             save_result_images_dir,
                             scores_n_arr=scores_n_arr)

        return scores_n_arr, sim_calc_time

    def similarities(self,
                     n=10,
                     save_results_dir=None,
                     save_result_images_dir=None,
                     pad_fn=None):
        """
        Function for creating the similarity matrix between embeddings in the dataset
        using cosine similarity.

        Parameters:
        -----------
        n : int
            Specifying the top n most similar images to store (and optionally
            save as images).
        save_results_dir : str
            Directory to save the search results tsv file.
        save_result_images_dir : str
            Directory to store search image results (top K images).
        """
        start_sim_calc = time()

        # initiate computation of cosine similarity
        cosine = nn.CosineSimilarity(dim=1).to(self.device)
        for image_path_i, embedding_i in tqdm(self.dataset.items()):
            scores = {}
            for image_path_j, embedding_j in self.dataset.items():
                sim = cosine(embedding_i, embedding_j)[0].item()
                scores[image_path_j] = sim
            scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
            self.sim_dict[image_path_i] = [
                (image_path_j, sim) for k, (image_path_j, sim) in enumerate(scores)
                if k < n
            ]

        sim_calc_time = time() - start_sim_calc

        # If there's a dir specified in save_result_images_dir, create result images
        if save_result_images_dir:
            for image_path in self.sim_dict.keys():
                self.save_images(image_path, save_result_images_dir, pad_fn=pad_fn)

        # Save the search results and scores as json file
        if save_results_dir:
            with open(os.path.join(save_results_dir, 'search_similarity_report.json'), 'w') as fout:
                json.dump(self.sim_dict, fout)

        return sim_calc_time

    def save_images(self, query_file, save_dir, scores_n_arr=None, pad_fn=None, corpus_dir=None):
        """Save similar images from similarity search.

        corpus_dir: directory where corpus proteogram images live. Required when
        embeddings were saved with filename-only keys (portable mode) so that
        result image paths can be reconstructed for display.
        """
        if not scores_n_arr:
            result_keys = [target for target, _ in self.sim_dict[query_file]]
            scores = ['']
            scores.extend([f'{score:.2f}' for _, score in self.sim_dict[query_file]])
        else:
            result_keys = [file for file, _ in scores_n_arr]
            scores = ['']
            scores.extend([f'{score:.2f}' for _, score in scores_n_arr])
        if corpus_dir:
            # Build a recursive filename→path index so subdirectory layouts are found.
            corpus_index = {
                os.path.basename(p): p
                for p in glob.glob(os.path.join(corpus_dir, '**', '*.jpg'), recursive=True)
            }
            corpus_index.update({
                os.path.basename(p): p
                for p in glob.glob(os.path.join(corpus_dir, '**', '*.png'), recursive=True)
            })
            result_paths = [
                corpus_index.get(os.path.basename(k), os.path.join(corpus_dir, os.path.basename(k)))
                for k in result_keys
            ]
        else:
            result_paths = result_keys

        # Filter to files that actually exist; warn about missing ones
        available = [(p, s) for p, s in zip(result_paths, scores[1:]) if os.path.isfile(p)]
        missing = [p for p in result_paths if not os.path.isfile(p)]
        if missing:
            print(f'Warning: {len(missing)} result image(s) not found on disk and will be skipped.')
            print(f'  Set proteograms_for_sim_dir in config.yml to the local corpus image directory.')
        result_paths = [p for p, _ in available]
        scores = [''] + [s for _, s in available]

        if not os.path.isfile(query_file):
            raise FileNotFoundError(
                f'Query proteogram image not found: {query_file}\n'
                f'Ensure the proteogram was saved successfully before calling save_images.')

        images_files = [query_file] + result_paths
        images = [pad_fn(Image.open(t)) if pad_fn else Image.open(t) for t in images_files]

        max_height = 1000
        total_width = max_height * len(images)
        font_size = max_height // 15
        path = os.path.dirname(__file__)
        font = ImageFont.truetype(os.path.join(path,'fonts','FreeSansBold.ttf'), font_size)

        new_im = Image.new('RGB', (total_width, max_height+70), color='white')

        x_offset = 0
        for i, im in enumerate(images):
            im = im.resize((max_height, max_height), Image.LANCZOS)
            new_im.paste(im, (x_offset,0))
            I1 = ImageDraw.Draw(new_im)
            if i > 0:
                I1.text((x_offset,max_height-5),
                        os.path.basename(images_files[i][:-4]) + f' = {scores[i]}', 
                        fill=(0, 0, 0),
                        font=font)
            else:
                # Don't need score since this is just the image query
                I1.text((x_offset,max_height-5),
                        os.path.basename(images_files[i][:-4]),
                        fill=(0, 0, 0),
                        font=font)
            x_offset+=max_height
            # x_offset += im.size[0]

        out_filename = save_dir + os.sep + \
            '.'.join(os.path.basename(query_file).split('.')[:-1]) + \
            '_top_sims.jpg'
        new_im.save(out_filename)

        return None
    
    def show_images(self, similar, target):
        self.display_img(target, "original")

        for k, v in similar.items():
            self.display_img(k, "similarity:" + str(v))

        return None

    def display_img(self, path, title):
        plt.imshow(Image.open(path))
        plt.axis("off")
        plt.title(title)
        plt.show()

        return

    def save_dataset(self):
        """
        Function to save a previously embedded image dataset to file

        Parameters:
        -----------
        path: str specifying the output folder to save the tensors to
        """

        # convert embeddings to dictionary
        data = {"model": self.architecture, "embeddings": self.dataset}

        torch.save(
            data, self.embed_file
        )  # need to update functionality for naming convention

    def load_dataset(self):
        """
        Function to save a previously embedded image dataset to file

        Parameters:
        -----------
        source: str specifying tensor.pt file to load previous embeddings
        """

        data = torch.load(self.embed_file)

        # assess that embedding nn matches currently initiated nn
        if data["model"] == self.architecture:
            self.dataset = data["embeddings"]
        else:
            raise AttributeError(
                f'NN architecture "{self.architecture}" does not match the '
                + f'"{data["model"]}" model used to generate saved embeddings.'
                + " Re-initiate Img2Vec with correct architecture and reload."
            )

    def plot_list(self, img_list, cluster_num):
        fig, axes = plt.subplots(math.ceil(len(img_list) / 2), 2)
        fig.suptitle(f"Cluster: {str(cluster_num)}")
        [ax.axis("off") for ax in axes.ravel()]

        for img, ax in zip(img_list, axes.ravel()):
            ax.imshow(Image.open(img))

        fig.tight_layout()

        return

    def display_clusters(self):
        for num in self.cluster_centers.keys():
            # print(f'Displaying cluster: {str(cluster_num)}')

            img_list = [k for k, v in self.image_clusters.items() if v == num]
            self.plot_list(img_list, num)

        return

    def cluster_dataset(self, nclusters, dist="euclidean", display=False):
        vecs = torch.stack(list(self.dataset.values())).squeeze()
        imgs = list(self.dataset.keys())
        np.random.seed(100)

        cluster_ids_x, cluster_centers = kmeans(
            X=vecs, num_clusters=nclusters, distance=dist, device=self.device
        )

        # assign clusters to images
        self.image_clusters = dict(zip(imgs, cluster_ids_x.tolist()))

        # store cluster centres
        cluster_num = list(range(0, len(cluster_centers)))
        self.cluster_centers = dict(zip(cluster_num, cluster_centers.tolist()))

        if display:
            self.display_clusters()

        return
