import boto3
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import torch
from joblib import Memory
from torch.utils.data import DataLoader, Dataset, random_split
from io import BytesIO
from PIL import Image
from typing import Tuple

bucket_name = 'guard-01'
endpoint_url='https://storage.yandexcloud.net'
cache_expiry = 86400 * 1000  # Cache expiry time in seconds (1000 days)
items_per_label = 1000

memory = Memory(location='object_cache')

class AstroS3Dataset(Dataset):
    def __init__(self,
                 aws_access_key_id,
                 aws_secret_access_key,
                 transform,
                 label_encoder,
                 exclude_label=['.ipynb_checkpoints/'],
                 cache_path='astro_data/objects_cache.json'):
        self.cache_path = cache_path
        self.s3_session = boto3.session.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key)
        self.s3_client = None
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

    def __initialize_s3_client(self):
        if self.s3_client is None:
            self.s3_client = self.s3_session.client(
                service_name='s3',
                endpoint_url=endpoint_url
            )

    def __get_object_keys_by_label(self):
        """ Get object keys for all folders, either from cache or from S3 """
        if _cache_exists_and_valid(self.cache_path):
            print("Loading object keys from cache...")
            return _load_cache(self.cache_path)
        
        print("Fetching object keys from S3...")
        self.__initialize_s3_client()
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
        response = self.s3_client.list_objects_v2(Bucket=bucket_name, Delimiter='/')
        return [prefix['Prefix'] for prefix in response.get('CommonPrefixes', [])]

    def __list_objects_by_prefix(self, prefix):
        """ List all objects under a given prefix (folder) """
        print(f"Listing objects under {prefix}")
        paginator = self.s3_client.get_paginator('list_objects_v2')
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
        self.__initialize_s3_client()
        object_data = self.__get_object_data(s3_key)
        try:
            with BytesIO(object_data) as object_stream:
                table = pd.read_csv(object_stream, skiprows=1, usecols=['jd', 'mag'], dtype={'jd':'float32', 'mag':'float32'})
                data=self.transform(table)
                label = self.encoded_labels[idx]
                del table
                return data, label
        except:
            print("S3 key", s3_key)
            raise

    def __get_object_data(self, s3_key):
        @memory.cache
        def get_object_data(s3_key):
            with self.s3_client.get_object(Bucket=bucket_name, Key=s3_key)['Body'] as body:
                return body.read()
        return get_object_data(s3_key)
    
    def __del__(self):
        if self.s3_client is not None:
            self.s3_client.close()

def get_jd_magn_graph_dataset(aws_access_key_id, aws_secret_access_key, label_encoder, cache_path=None):
    transform = transforms.Compose([
        transforms.Lambda(_get_jd_magn_graph),
        transforms.ToTensor()
    ])
    return AstroS3Dataset(aws_access_key_id, aws_secret_access_key, transform, label_encoder, cache_path=cache_path)

def get_jd_magn_1d_dataset(aws_access_key_id, aws_secret_access_key, label_encoder, cache_path=None):
    transform = transforms.Compose([
        transforms.Lambda(_get_jd_magn_1d),
        #transforms.ToTensor()
    ])
    return AstroS3Dataset(aws_access_key_id, aws_secret_access_key, transform, label_encoder, cache_path=cache_path)

def get_train_test(dataset: Dataset, split_ratio=0.7, batch_size=32) -> Tuple[DataLoader, DataLoader]:
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = _get_dataloader(batch_size, train_dataset)
    test_loader = _get_dataloader(batch_size, test_dataset)

    return train_loader, test_loader

def _get_dataloader(batch_size, train_dataset):
    return DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        prefetch_factor=None)

def _get_jd_magn_graph(df: pd.DataFrame) -> Image.Image:
    x = df['jd']
    y = df['mag']

    fig, ax = plt.subplots()
    ax.plot(x, y, color='black')    # Monochrome line (directly in grayscale)
    ax.axis('off')                  # Turn off axes for clean bitmap

    # Save the figure to a buffer in grayscale
    with BytesIO() as buf:
        fig.tight_layout(pad=0)
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        # Open the image directly as grayscale
        image = Image.open(buf).convert('1')  # Directly to monochrome
        return image

def _get_jd_magn_1d(df: pd.DataFrame, width: int = 400):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how='any', inplace=True)
    x_vals = df['jd'].to_numpy()
    y_vals = df['mag'].to_numpy()

    if len(x_vals) < 2:
        raise ValueError("Need at least two data points.")

    # Normalize x to [0, 1] and interpolate y over fixed-width samples
    x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
    y_interp = np.interp(np.linspace(0, 1, width), x_norm, y_vals)

    if (np.isnan(y_interp).any() or np.isinf(y_interp).any() or
        np.isclose(y_interp.max(), y_interp.min())):
        raise ValueError("Bad data: NaNs, Infs, or flat signal")
    
    y_norm = (y_interp - y_interp.min()) / (y_interp.max() - y_interp.min())

    return torch.tensor(y_norm, dtype=torch.float32).unsqueeze(0)

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
