import boto3
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import time
import torch
import random
from joblib import Memory
from torch.utils.data import DataLoader, Dataset, random_split
from io import BytesIO
from PIL import Image
from typing import Tuple

bucket_name = 'guard-01'
endpoint_url='https://storage.yandexcloud.net'
cache_expiry = 86400 * 1000  # Cache expiry time in seconds (1000 days)

memory = Memory(location='object_cache', verbose=0)

class AstroS3Dataset(Dataset):
    def __init__(self,
                 aws_access_key_id,
                 aws_secret_access_key,
                 transform,
                 label_encoder,
                 exclude_label=['.ipynb_checkpoints/', 'Dataformodel/'],
                 cache_path='astro_data/objects_cache.json',
                 items_per_label=1000):
        self.cache_path = cache_path
        self.s3_session = boto3.session.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key)
        self.s3_client = None
        keys_by_label = self.__get_object_keys_by_label()
        keys_by_label = {
            key: [s for s in values
                  if s.endswith('.csv')
                  and not s.endswith('index.csv')
                  and '.ipynb_checkpoints' not in s
                  and not s.endswith('128849309969.csv')
                  and not s.endswith('180389457397.csv')]
            for key, values in keys_by_label.items()
        }
        total_labels = list(keys_by_label.keys())
        for el in exclude_label:
            total_labels.remove(el)
        self.s3_keys  = []
        labels   = []
        for label in total_labels:
            elements_count = len(keys_by_label[label])
            print('items per label', items_per_label)
            if elements_count < items_per_label:
                print(f"WARNING: The limit for items per class is set to {items_per_label}, but there are only {elements_count} elements in {label}.")
                self.s3_keys += (keys_by_label[label])
                labels += ([label.rstrip('/')] * elements_count)
            else:
                self.s3_keys += (random.sample(keys_by_label[label], items_per_label))
                labels += ([label.rstrip('/')] * items_per_label)
        self.transform = transform
        self.label_encoder = label_encoder
        self.encoded_labels = label_encoder.fit_transform(labels)
    
    def get_class_indices(self, label: str):
        class_val = self.label_encoder.transform([label])[0]
        indices = [i for i, val in enumerate(self.encoded_labels) if val == class_val]
        return indices
    
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
        try:
            object_data = self.__get_object_data(s3_key)
            with BytesIO(object_data) as object_stream:
                table = pd.read_csv(object_stream, skiprows=1, usecols=['jd', 'mag', 'mag_err', 'phot_filter'], dtype={'jd':'float32', 'mag':'float32', 'mag_err':'float32', 'phot_filter': 'str'})
                data=self.transform(table)
                label = self.encoded_labels[idx]
                del table
                return data, label
        except:
            print("Could not load or parse table. S3 key:", s3_key)
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

class AstroFileDataset(Dataset):
    def __init__(self,
                 root_dir,
                 transform,
                 label_encoder,
                 items_per_label=1000):
        """
        Args:
            root_dir (str): Root directory containing labeled folders with CSV files.
            transform (callable): Function to transform the loaded DataFrame into model-ready format.
            label_encoder (LabelEncoder): Optional external label encoder. If None, a new one is created.
            items_per_label (int): Max number of items to load per label.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.file_paths = []
        labels = []

        total_labels = [folder for folder in os.listdir(root_dir)
                        if os.path.isdir(os.path.join(root_dir, folder))]

        for label in total_labels:
            label_path = os.path.join(root_dir, label)
            all_files = [os.path.join(label_path, f) for f in os.listdir(label_path)
                         if f.endswith('.csv')]
            n_files = len(all_files)
            if n_files < items_per_label:
                print(f"WARNING: Only {n_files} items in label '{label}', less than requested {items_per_label}")
                selected_files = all_files
            else:
                selected_files = all_files[:items_per_label]
            self.file_paths.extend(selected_files)
            labels.extend([label] * len(selected_files))

        self.encoded_labels = label_encoder.transform(labels)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            df = pd.read_csv(file_path)
            data = self.transform(df)
            label = self.encoded_labels[idx]
            del df
            return data, label
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            raise

def get_jd_magn_1d_dataset(aws_access_key_id, aws_secret_access_key, label_encoder, cache_path=None, items_per_label=None, vector_length=400, use_random_filter=False):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: _get_jd_magn_1d(x, width=vector_length, use_random_filter=use_random_filter))
    ])
    return AstroS3Dataset(aws_access_key_id, aws_secret_access_key, transform, label_encoder, cache_path=cache_path, items_per_label=items_per_label)

def get_jd_magn_magerr_1d_dataset(aws_access_key_id, aws_secret_access_key, label_encoder, cache_path=None, items_per_label=None, vector_length=400, use_random_filter=False):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: _get_jd_magn_magerr_1d(x, width=vector_length, use_random_filter=use_random_filter))
    ])
    return AstroS3Dataset(aws_access_key_id, aws_secret_access_key, transform, label_encoder, cache_path=cache_path, items_per_label=items_per_label)

def get_train_test(dataset: Dataset, split_ratio=0.7, batch_size=32) -> Tuple[DataLoader, DataLoader]:
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = _get_dataloader(batch_size, train_dataset)
    test_loader = None
    if (split_ratio < 1.0):
        test_loader = _get_dataloader(batch_size, test_dataset)

    return train_loader, test_loader

def _get_dataloader(batch_size, dataset):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        prefetch_factor=2)

def _get_jd_magn_1d(df: pd.DataFrame, width: int = 400, use_random_filter=False):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how='any', inplace=True)
    if use_random_filter:
        available_filters = df['phot_filter'].unique()
        if len(available_filters) == 0:
            raise ValueError("No photometric filters found.")

        chosen_filter = random.choice(available_filters)
        df = df[df['phot_filter'] == chosen_filter]
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

def _get_jd_magn_magerr_1d(df: pd.DataFrame, width: int = 400, use_random_filter=False):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(how='any', inplace=True)

    if use_random_filter:
        available_filters = df['phot_filter'].unique()
        if len(available_filters) == 0:
            raise ValueError("No photometric filters found.")
        chosen_filter = random.choice(available_filters)
        df = df[df['phot_filter'] == chosen_filter]

    x_vals = df['jd'].to_numpy()
    mag_vals = df['mag'].to_numpy()
    magerr_vals = df['mag_err'].to_numpy()

    if len(x_vals) < 2:
        raise ValueError("Need at least two data points.")

    # Normalize x to [0, 1] and interpolate y values over fixed-width samples
    x_norm = (x_vals - x_vals.min()) / (x_vals.max() - x_vals.min())
    interp_points = np.linspace(0, 1, width)

    mag_interp = np.interp(interp_points, x_norm, mag_vals)
    magerr_interp = np.interp(interp_points, x_norm, magerr_vals)

    # Sanity checks
    if (
        np.isnan(mag_interp).any() or np.isinf(mag_interp).any() or
        np.isclose(mag_interp.max(), mag_interp.min()) #or
        # np.isnan(magerr_interp).any() or np.isinf(magerr_interp).any() or
        # np.isclose(magerr_interp.max(), magerr_interp.min())
    ):
        raise ValueError("Bad data: NaNs, Infs, or flat signal")

    # Normalize mag
    mag_norm = (mag_interp - mag_interp.min()) / (mag_interp.max() - mag_interp.min())

    # Normalize and invert magerr, or fallback to zeros
    if (
        np.isnan(magerr_interp).any() or np.isinf(magerr_interp).any() or
        np.isclose(magerr_interp.max(), magerr_interp.min())
    ):
        magerr_inverted = np.zeros_like(magerr_interp)
    else:
        magerr_norm = (magerr_interp - magerr_interp.min()) / (magerr_interp.max() - magerr_interp.min())
        magerr_inverted = 1.0 - magerr_norm

    stacked = np.stack([mag_norm, magerr_inverted], axis=0)  # Shape: (2, width)
    return torch.tensor(stacked, dtype=torch.float32)

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
