import torch

print("Checking for GPU...")
if torch.cuda.is_available():
    print(f"Success! GPU found: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU found. Check your Docker/Lightning setup.")