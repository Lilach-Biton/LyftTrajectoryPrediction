## Not Tested YET
import argparse
from utils import load_config, get_model_class
from dataset import load_datasets
from trainer import Trainer
import torch


def main(config_path, model_name, exp_name):
    cfg = load_config(config_path)
    ModelClass = get_model_class(model_name)

    train_loader, val_loader = load_datasets(cfg)

    model = ModelClass(
        in_channels=cfg["model_params"]["history_num_frames"] * 2 + 3,
        num_targets=cfg["model_params"]["future_num_frames"] * 2,
        num_modes=cfg["model_params"]["num_modes"]
    )

    trainer = Trainer(cfg, model, train_loader, val_loader, exp_name)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--model", type=str, required=True, help="Model file name in models/ without .py")
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()

    main(args.config, args.model, args.exp_name)
