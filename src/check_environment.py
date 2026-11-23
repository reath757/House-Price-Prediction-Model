
import sys
import platform
import importlib.metadata

print("=" * 50)
print("ZIMBABWE HOUSE PRICE RESEARCH")
print("ENVIRONMENT VERIFICATION")
print("=" * 50)

print("Python:", sys.version)
print("Operating system:", platform.platform())

packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "torch",
    "torchvision",
    "transformers",
    "sentence-transformers",
    "xgboost",
    "jupyter",
    "shap"
]

for package in packages:
    try:
        version = importlib.metadata.version(package)
        print(f"{package}: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{package}: NOT INSTALLED")

try:
    import torch

    print("\nPyTorch information")
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Running on CPU")

except Exception as error:
    print("PyTorch error:", error)

print("\nEnvironment verification complete.")
