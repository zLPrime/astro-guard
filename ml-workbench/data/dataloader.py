import boto3
import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, Dataset
from io import BytesIO
from PIL import Image

bucket_name = 'guard-01'
endpoint_url='https://storage.yandexcloud.net'
cache_file = 'objects_cache.json'
cache_expiry = 86400 * 1000  # Cache expiry time in seconds (1000 days)
items_per_label = 50000

class AstroS3Dataset(Dataset):
    def __init__(self, aws_access_key_id, aws_secret_access_key, transform, exclude_label=['.ipynb_checkpoints/']):
        session = boto3.session.Session()
        self.s3 = session.client(
            service_name='s3',
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        keys_by_label = get_object_keys_by_label()
        labels = list(keys_by_label.keys())
        for el in exclude_label:
            labels.remove(el)
        self.s3_keys  = []
        self.labels   = []
        for label in labels:
            elements_count = len(keys_by_label[label])
            if elements_count < items_per_label:
                print(f"WARNING: The limit for items per class is set to {items_per_label}, but there are only {elements_count} elements in {label}.")
                self.s3_keys += (keys_by_label[label])
                self.labels += ([label] * elements_count)
            else:
                self.s3_keys += (keys_by_label[label][:items_per_label])
                self.labels += ([label] * items_per_label)
        self.transform = transform
    
    def get_folders(self):
        """ Get all top-level folders in the bucket """
        response = self.s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
        return [prefix['Prefix'] for prefix in response.get('CommonPrefixes', [])]

    def __len__(self):
        return len(self.s3_keys)
    
    def __getitem__(self, idx):
        s3_key = self.s3_keys[idx]
        assert isinstance(s3_key, str)
        get_object_response = self.s3.get_object(Bucket=bucket_name, Key=s3_key)
        table = pd.read_csv(get_object_response['Body'], skiprows=1)
        data=self.transform(table)
        print(type(data))
        assert isinstance(data, Image.Image)
        label = self.labels[idx]
        return data, label

def get_jd_magn_graph_dataloader(aws_access_key_id, aws_secret_access_key, batch_size=32, shuffle=True):
    dataset = AstroS3Dataset(aws_access_key_id, aws_secret_access_key, get_jd_magn_graph)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_jd_magn_graph(df: pd.DataFrame) -> Image.Image:
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
    print(type(image))
    return image

def cache_exists_and_valid():
    """ Check if the cache file exists and is still valid """
    if os.path.exists(cache_file):
        cache_age = time.time() - os.path.getmtime(cache_file)
        return cache_age < cache_expiry
    return False

def load_cache():
    """ Load the object keys from cache """
    with open(cache_file, 'r') as f:
        return json.load(f)

def save_cache(cache_data):
    """ Save the object keys to cache """
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

def get_object_keys_by_label():
    """ Get object keys for all folders, either from cache or from S3 """
    if cache_exists_and_valid():
        print("Loading object keys from cache...")
        return load_cache()
    
    print("Fetching object keys from S3...")
    folders = get_folders()
    cache_data = {}
    
    print("Collect object keys for each folder...")
    for folder in folders:
        cache_data[folder] = list_objects_by_prefix(folder)
    
    # Save to cache
    save_cache(cache_data)
    return cache_data

def list_objects_by_prefix(prefix):
    """ List all objects under a given prefix (folder) """
    print(f"Listing s under the prefix {prefix}")
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
    objects = []
    
    n_objects = 0
    for i, page in enumerate(pages):
        if 'Contents' in page:
            n_objects += len(page['Contents'])
            if i % 50 == 0:
              print(f"Page {i}. {n_objects} items processed.")
            for obj in page['Contents']:
                objects.append(obj['Key'])
    
    return objects