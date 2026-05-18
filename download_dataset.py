import urllib.request
import os

# URL for the tiny Shakespeare dataset
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Output file path
output_file = "shakespeare_dataset.txt"

print(f"Downloading tiny Shakespeare dataset from {url}...")
try:
    urllib.request.urlretrieve(url, output_file)
    file_size = os.path.getsize(output_file)
    print(f"✓ Dataset downloaded successfully!")
    print(f"✓ Saved to: {output_file}")
    print(f"✓ File size: {file_size:,} bytes")
except Exception as e:
    print(f"✗ Error downloading dataset: {e}")
