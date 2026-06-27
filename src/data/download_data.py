import kagglehub
import shutil
from pathlib import Path
from src.config import RAW_DATA_DIR

def main():
    # Download latest version
    path = kagglehub.dataset_download("tobiasbueck/multilingual-customer-support-tickets")

    print("Path to dataset files:", path)

    # Define destination directory
    dest_dir = RAW_DATA_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Copying files to {dest_dir}...")
    source_path = Path(path)
    for item in source_path.iterdir():
        if item.is_file():
            shutil.copy2(item, dest_dir / item.name)
        elif item.is_dir():
            shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)

    print("Done!")

if __name__ == "__main__":
    main()
