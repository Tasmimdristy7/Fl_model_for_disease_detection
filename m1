import numpy as np
import pandas as pd
import random
import cv2
import os
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import expand_dims
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input, Lambda
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K
pip install pillow
from google.colab import drive
drive.mount('/content/gdrive',force_remount=True)
df = pd.read_csv('/content/gdrive/MyDrive/fl/Data_Entry_2017.csv')
df.head(10)
from PIL import Image
import os, os.path

imgs = []
path = "/content/drive/MyDrive/fl/images"
valid_images = [".png",]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(Image.open(os.path.join(path,f)))
    
 def showImage(imgs):
 plt.imshow(np.array(imgs) / 255)
    
 def loadImage(imgs):
 return Image.open(path).convert('RGB')
 
 showImage(loadImage(imgs))
 
 classes = [
    'Atelectasis', 
    'Consolidation', 
    'Infiltration', 
    'Pneumothorax', 
    'Edema', 
    'Emphysema', 
    'Fibrosis', 
    'Effusion', 
    'Pneumonia', 
    'Pleural_thickening', 
    'Cardiomegaly', 
    'Nodule', 
    'Mass', 
    'Hernia', 
    'No Finding'
]

class ImageDataset(Dataset):
    def __init__(self, data, transforms):
        self.image_paths = [imgs for f in data[0]]
        self.labels = data[1]
        self.transforms = transforms
        
    def __len__(self):
        return len(data[0])
    
    def __getitem__(self, idx):
        image = self.transforms(loadImage(self.image_paths[idx]))
        target = torch.tensor([int(cls in self.labels[idx]) for cls in classes], dtype=torch.float32)
        return (image, target)
        
data = (df.iloc[:5000, 0], [df.iloc[i, 1].split('|') for i in range(5000)])

dataset = ImageDataset(data, trns.Compose([
    trns.Resize((240, 240)),
    trns.ToTensor(), 
    trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],inplace=True)
]))

print(dataset[1][1])
plt.imshow(dataset[1][0].permute((1, 2, 0)))

train_dataset, validation_dataset = random_split(dataset, [int(len(dataset) * 0.85), 
            len(dataset) - int(len(dataset) * 0.85)])
train_dataset_size = len(train_dataset)
validation_dataset_size = len(validation_dataset)

train_dataset_size, validation_dataset_size


    
