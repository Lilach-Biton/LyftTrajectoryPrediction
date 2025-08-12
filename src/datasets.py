### Not tested yet

import os
from l5kit.data import LocalDataManager
from l5kit.dataset import EgoDataset
from l5kit.rasterization import build_rasterizer
from torch.utils.data import DataLoader


def load_datasets(cfg, data_path):
    # set env variable for data
    os.environ["L5KIT_DATA_FOLDER"] = data_path
    dm = LocalDataManager(None)
    rasterizer = build_rasterizer(cfg, dm)

    train_dataset = EgoDataset(cfg, dm, rasterizer, cfg["train_data_loader"]["key"])
    val_dataset = EgoDataset(cfg, dm, rasterizer, cfg["val_data_loader"]["key"])

    train_loader = DataLoader(train_dataset, **cfg["train_data_loader"])
    val_loader = DataLoader(val_dataset, **cfg["val_data_loader"])

    return train_loader, val_loader
