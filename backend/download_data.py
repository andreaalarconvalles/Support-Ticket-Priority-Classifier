import kagglehub
import shutil
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("muqaddasejaz/customer-support-ticket-dataset")

print("Path to dataset files:", path)

# Define destination directory (relative to this script, so it works from any cwd)
dest_dir = Path(__file__).parent / "data" / "raw"
dest_dir.mkdir(parents=True, exist_ok=True)

print(f"Copying files to {dest_dir}...")
source_path = Path(path)
for item in source_path.iterdir():
    if item.is_file():
        shutil.copy2(item, dest_dir / item.name)
    elif item.is_dir():
        shutil.copytree(item, dest_dir / item.name, dirs_exist_ok=True)

print("Done!")