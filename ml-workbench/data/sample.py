import boto3
import random
import json
import os
import time

# Configuration
bucket_name = 'guard-01'
from google.colab import userdata
sample_size = 5  # Number of samples per folder
cache_file = 'objects_cache.json'
cache_expiry = 86400 * 1000  # Cache expiry time in seconds (1000 days)

# Initialize S3 client
session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id=userdata.get('aws_access_key_id'),
    aws_secret_access_key=userdata.get('aws_secret_access_key')
)

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

def get_folders():
    """ Get all top-level folders in the bucket """
    response = s3.list_objects_v2(Bucket=bucket_name, Delimiter='/')
    return [prefix['Prefix'] for prefix in response.get('CommonPrefixes', [])]

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

def get_object_keys():
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

def sample_from_folder(objects, sample_size):
    """ Sample a given number of objects from a list of keys """
    if len(objects) <= sample_size:
        return objects  # Return all if fewer than the sample size
    return random.sample(objects, sample_size)

def main():
    object_keys = get_object_keys()
    samples = {}
    
    # Sample from each folder's objects
    for folder, keys in object_keys.items():
        samples[folder] = sample_from_folder(keys, sample_size)
    
    # Print the samples
    for folder, objects in samples.items():
        print(f"\nSamples from {folder}:")
        for obj in objects:
            print(obj)

if __name__ == '__main__':
    main()