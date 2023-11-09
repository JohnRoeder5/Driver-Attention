import requests
import os

# Function to download files from a URL
def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded: {local_filename}")

# Create directories to store the files if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('labels', exist_ok=True)

# URLs for GoogLeNet model files and class labels file
googlenet_deploy_url = 'https://raw.githubusercontent.com/BVLC/caffe/master/models/bvlc_googlenet/deploy.prototxt'
googlenet_caffemodel_url = 'http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
imagenet_labels_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'

# Download GoogLeNet model files
download_file(googlenet_deploy_url, 'models/deploy.prototxt')
download_file(googlenet_caffemodel_url, 'models/bvlc_googlenet.caffemodel')

# Download class labels file
download_file(imagenet_labels_url, 'labels/imagenet-simple-labels.json')
