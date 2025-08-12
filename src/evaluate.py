### Not tested yet

import argparse
import torch
from utils import load_config, get_model_class
from dataset import load_datasets


def main(config_path, model_name, checkpoint):
    cfg = load_config(config_path)
    ModelClass = get_model_class(model_name)

    _, val_loader = load_datasets(cfg)

    model = ModelClass(
        in_channels=cfg["model_params"]["history_num_frames"] * 2 + 3,
        num_targets=cfg["model_params"]["future_num_frames"] * 2,
        num_modes=cfg["model_params"]["num_modes"]
    )

    model.load_state_dict(torch.load(checkpoint))
    model.eval()

    # TODO: Implement evaluation
    print("Evaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/baseline.yaml")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    main(args.config, args.model, args.checkpoint)
