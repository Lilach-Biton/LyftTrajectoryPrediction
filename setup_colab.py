# setup_colab.py
import os
import subprocess

def download_data_if_needed(data_dir, onedrive_url):
    if os.path.exists(data_dir) and os.listdir(data_dir):
        print(f"Data directory '{data_dir}' already exists and is not empty. Skipping download.")
        return
    else:
        print(f"Data directory '{data_dir}' not found or empty. Downloading dataset...")

    zip_path = "/tmp/dataset.zip"
    
    # Download dataset zip from OneDrive with direct download link
    # You might need to adjust the URL to include ?download=1 at the end
    download_url = onedrive_url
    if not download_url.endswith("?download=1"):
        download_url += "?download=1"
    
    # Use wget to download dataset
    cmd = f"wget -O {zip_path} '{download_url}'"
    print(f"Running command: {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    
    # Unzip dataset
    print("Extracting dataset...")
    subprocess.run(f"unzip -o {zip_path} -d {data_dir}", shell=True, check=True)
    
    print(f"Dataset downloaded and extracted to '{data_dir}'")

def main():
    repo_name = "LyftTrajectoryPrediction"
    repo_url = f"https://github.com/Lilach-Biton/{repo_name}.git"

    if not os.path.exists(repo_name):
        print(f"Cloning repo from {repo_url}...")
        os.system(f"git clone {repo_url}")
    else:
        print(f"Repo '{repo_name}' already exists. Skipping clone.")

    os.chdir(repo_name)
    print(f"Changed directory to {os.getcwd()}")

    if os.path.exists("requirements.txt"):
        print("Installing additional requirements from requirements.txt...")
        os.system("pip install -r requirements.txt")
    else:
        print("No requirements.txt found, skipping.")

    # Checkpoint directory (local in Colab VM)
    checkpoint_dir = "./checkpoints"
    print(f"Creating checkpoint directory at {checkpoint_dir} if it doesn't exist...")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Dataset directory (local in Colab VM)
    data_dir = "./lyft_data"

    data_url = "https://technionmail-my.sharepoint.com/:u:/g/personal/lilach_biton_campus_technion_ac_il/EYEffRkWayVBrOz1VhlrnF0BME8qdX9l23LGALs23hO4pw?e=d1Akxb&download=1"

    download_data_if_needed(data_dir, data_url)

if __name__ == "__main__":
    main()

