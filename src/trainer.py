import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, cfg, model, device, train_loader, exp_name):
        self.cfg = cfg
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.exp_dir = os.path.join("experiments", exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg["train_params"]["learning_rate"])

    def train(self):
        # for epoch in range(self.cfg["train_params"]["max_num_epochs"]):
        #     self.model.train()
        #     for batch in self.train_loader:
        #         self.optimizer.zero_grad()
        #         outputs = self.model(batch["image"].to(self.device))
        #         loss = self.criterion(outputs, batch["target_positions"].to(self.device))
        #         loss.backward()
        #         self.optimizer.step()
        #     torch.save(self.model.state_dict(), os.path.join(self.exp_dir, f"epoch_{epoch}.pth"))
        tr_it = iter(self.train_loader)
        progress_bar = tqdm(range(self.cfg["train_params"]["max_num_steps"]))
        losses_train = []
        for _ in progress_bar:
            try:
                data = next(tr_it)
            except StopIteration:
                tr_it = iter(self.train_loader)
                data = next(tr_it)
            self.model.train()
            torch.set_grad_enabled(True)
            loss, _ = self.model.forward_pass(data, self.device, self.criterion)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            losses_train.append(loss.item())
            progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")
        torch.save(self.model.state_dict(), os.path.join(self.exp_dir, f"epoch_0.pth"))
        return losses_train

