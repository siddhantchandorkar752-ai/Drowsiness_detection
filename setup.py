import os
import shutil
import random
import zipfile
import subprocess
import sys
from pathlib import Path


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def check_kaggle():
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle...")
        install("kaggle")


def check_kaggle_token():
    kaggle_path = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_path):
        print("\nkaggle.json not found!")
        print("Steps:")
        print("1. Go to https://www.kaggle.com/settings")
        print("2. Scroll to API section")
        print("3. Click Create New Token")
        print("4. Move kaggle.json to:", kaggle_path)
        input("\nPress Enter after placing kaggle.json...")


def download_dataset():
    print("\nDownloading MRL Eye Dataset...")
    os.makedirs("data", exist_ok=True)
    os.system("kaggle datasets download -d prasadvpatil/mrl-dataset -p data/")

    print("Extracting...")
    with zipfile.ZipFile("data/mrl-dataset.zip", "r") as z:
        z.extractall("data/")
    print("Dataset ready.")


def organize_folders():
    print("\nOrganizing train/valid folders...")

    for split in ["train", "valid"]:
        for cls in ["Closed", "Open"]:
            os.makedirs(f"data/{split}/{cls}", exist_ok=True)

    class_map = {"Closed": "Closed_Eyes", "Open": "Open_Eyes"}

    for cls, folder in class_map.items():
        src = Path(f"data/{folder}")

        if not src.exists():
            for alt in Path("data").iterdir():
                if alt.is_dir() and ("losed" in alt.name or "pen" in alt.name):
                    src = alt
                    break

        files = list(src.glob("*.png")) + list(src.glob("*.jpg"))
        random.shuffle(files)
        split_idx = int(len(files) * 0.8)

        for f in files[:split_idx]:
            shutil.copy(f, f"data/train/{cls}/{f.name}")

        for f in files[split_idx:]:
            shutil.copy(f, f"data/valid/{cls}/{f.name}")

    print("Train/Valid split done.")


def train_model():
    print("\nTraining model...")
    os.makedirs("models", exist_ok=True)
    os.system("python model.py")
    print("Model saved to models/cnnCat2.h5")


def main():
    check_kaggle()
    check_kaggle_token()
    download_dataset()
    organize_folders()
    train_model()
    print("\nSetup complete. Now run: python app.py")


if __name__ == "__main__":
    main()