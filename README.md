# Motion Prediction for Autonomous Vehicles - Lyft Level 5 Benchmark

This project aims to predict the future motion trajectories of traffic agents around the AV, such as cars, cyclists, and pedestrians, in complex urban environments, using the Lyft Level 5 dataset. This project is built on **[L5Kit](https://github.com/woven-planet/l5kit)**, a library designed for autonomous driving motion prediction.

---

## How to Run

### 1️⃣ Clone the repository

```bash
git clone --recurse-submodules https://github.com/Lilach-Biton/LyftTrajectoryPrediction.git
cd LyftTrajectoryPrediction
````

If you didn't clone submodules at first:

```bash
git submodule update --init --recursive
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Explore & train

* **Dataset Setup for Training:**

The full dataset couldn't be uploaded due to storage limitations; therefore, for the training process you are requested to manually download the dataset:

1. Download the dataset ZIP file from the Kagle competition:
   [Download dataset](https://www.kaggle.com/competitions/lyft-motion-prediction-autonomous-vehicles/data)

2. Extract the contents into the project folder under:
   `./data`

Make sure the extracted data is accessible at this path before running any scripts.

* **Running on Google Colab:**

For Google Colab users, simply run:
 NOT WORKING YET - TODO
```python
!python setup_colab.py
```

This script will clone the repo, install dependencies, download and extract the dataset automatically, and create necessary directories.

---

## License & Credits

This project uses **L5Kit**, licensed under **Apache 2.0**.
This project (in `LyftTrajectoryPrediction/`) is also shared under the same license unless stated otherwise.

---

## Contact & Issues

If you have any questions or find issues, please open an issue on GitHub or contact me directly.
