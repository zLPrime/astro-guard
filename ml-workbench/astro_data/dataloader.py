import boto3
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
from torch.utils.data import DataLoader, Dataset, random_split
from io import BytesIO
from PIL import Image
from typing import Tuple

bucket_name = 'guard-01'
endpoint_url='https://storage.yandexcloud.net'
cache_expiry = 86400 * 1000  # Cache expiry time in seconds (1000 days)
items_per_label = 1000

class AstroS3Dataset(Dataset):
    def __init__(self,
                 aws_access_key_id,
                 aws_secret_access_key,
                 transform,
                 label_encoder,
                 exclude_label=['.ipynb_checkpoints/'],
                 cache_path='astro_data/objects_cache.json'):
        self.cache_path = cache_path
        session = boto3.session.Session()
        self.s3 = session.client(
            service_name='s3',
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        keys_by_label = self.__get_object_keys_by_label()
        keys_by_label = {
            key: [s for s in values if s.endswith('.csv')]
            for key, values in keys_by_label.items()
        }
        total_labels = list(keys_by_label.keys())
        for el in exclude_label:
            total_labels.remove(el)
        self.s3_keys  = []
        labels   = []
        for label in total_labels:
            elements_count = len(keys_by_label[label])
            if elements_count < items_per_label:
                print(f"WARNING: The limit for items per class is set to {items_per_label}, but there are only {elements_count} elements in {label}.")
                self.s3_keys += (keys_by_label[label])
                labels += ([label] * elements_count)
            else:
                self.s3_keys += (keys_by_label[label][:items_per_label])
                labels += ([label] * items_per_label)
        self.transform = transform
        self.label_encoder = label_encoder
        self.encoded_labels = label_encoder.fit_transform(labels)

    def __get_object_keys_by_label(self):
        """ Get object keys for all folders, either from cache or from S3 """
        if _cache_exists_and_valid(self.cache_path):
            print("Loading object keys from cache...")
            return _load_cache(self.cache_path)
        
        print("Fetching object keys from S3...")
        folders = self.__get_folders()
        cache_data = {}
        
        print("Collect object keys for each folder...")
        for folder in folders:
            cache_data[folder] = self.__list_objects_by_prefix(folder)
        
        # Save to cache
        _save_cache(cache_data, self.cache_path)
        return cache_data
    
    def __get_folders(self):
        """ Get all top-level folders in the bucket """
        response = self.s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
        return [prefix['Prefix'] for prefix in response.get('CommonPrefixes', [])]

    def __list_objects_by_prefix(self, prefix):
        """ List all objects under a given prefix (folder) """
        print(f"Listing objects under {prefix}")
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
        objects = []
        
        n_objects = 0
        for i, page in enumerate(pages):
            if 'Contents' in page:
                if i % 50 == 0:
                    print(f"Page {i}. {n_objects} items processed.")
                n_objects += len(page['Contents'])
                for obj in page['Contents']:
                    objects.append(obj['Key'])
        
        return objects

    def __len__(self):
        return len(self.s3_keys)
    
    def __getitem__(self, idx):
        s3_key = self.s3_keys[idx]
        assert isinstance(s3_key, str)
        get_object_response = self.s3.get_object(Bucket=bucket_name, Key=s3_key)
        try:
          table = pd.read_csv(get_object_response['Body'], skiprows=1)
          data=self.transform(table)
          label = self.encoded_labels[idx]
          return data, label
        except:
          print("S3 key", s3_key)
          raise

def get_jd_magn_graph_dataset(aws_access_key_id, aws_secret_access_key, label_encoder, cache_path=None):
    transform = transforms.Compose([
        transforms.Lambda(_get_jd_magn_graph),
        transforms.ToTensor()
    ])
    return AstroS3Dataset(aws_access_key_id, aws_secret_access_key, transform, label_encoder, cache_path=cache_path)

def get_train_test(dataset: Dataset, split_ratio=0.7, batch_size=32) -> Tuple[DataLoader, DataLoader]:
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def _get_jd_magn_graph(df: pd.DataFrame) -> Image.Image:
    x = df['jd']
    y = df['mag']

    fig, ax = plt.subplots()
    ax.plot(x, y, color='black')    # Monochrome line (directly in grayscale)
    ax.axis('off')                  # Turn off axes for clean bitmap

    # Save the figure to a buffer in grayscale
    buf = BytesIO()
    fig.tight_layout(pad=0)
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)

    # Open the image directly as grayscale
    image = Image.open(buf).convert('1')  # Directly to monochrome
    return image

def _cache_exists_and_valid(cache_path):
    """ Check if the cache file exists and is still valid """
    if os.path.exists(cache_path):
        cache_age = time.time() - os.path.getmtime(cache_path)
        return cache_age < cache_expiry
    return False

def _load_cache(cache_path):
    """ Load the object keys from cache """
    with open(cache_path, 'r') as f:
        return json.load(f)

def _save_cache(cache_data, cache_path):
    """ Save the object keys to cache """
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
