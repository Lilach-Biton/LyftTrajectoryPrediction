import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from src.utils import pytorch_neg_multi_log_likelihood_single
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, cfg, model, device, train_loader, exp_name):
        self.cfg = cfg
        self.device = device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.exp_dir = os.path.join("experiments", exp_name)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(self.exp_dir, "logs"))

        # Loss and Optimizer
        self.criterion = nn.MSELoss()
        # self.criterion = pytorch_neg_multi_log_likelihood_single
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg["train_params"]["learning_rate"])
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2, patience=1, threshold=0.001)

        # Epochs
        self.num_epochs = cfg["train_params"]["num_epochs"]
        num_steps_per_epoch = len(self.train_loader.dataset)
        self.max_num_steps = min(num_steps_per_epoch, cfg["train_params"]["max_num_steps"])
        self.max_num_steps_val = max(1, int(self.max_num_steps / 10.0))

    def train_and_validate(self, val_loader, checkpoint=None):

        progress_bar = tqdm(range(self.max_num_steps))
        losses_train = []
        losses_train_epoch = []
        losses_val_epoch = []
        global_batch_idx = 0
        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch + 1} - Starting")
            progress_bar.reset()
            self.model.train()
            running_loss = 0.0
            torch.set_grad_enabled(True)
            
            for idx, data in enumerate(self.train_loader):
                if idx == self.max_num_steps:
                    break
                
                # Forward pass
                loss, _ = self.model.forward_pass(data, self.device, self.criterion)
                self.writer.add_scalar("Training Loss | Batch", loss.item(), global_batch_idx)
                global_batch_idx += 1
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses_train.append(loss.item())
                running_loss += loss.item()

                progress_bar.update()
                progress_bar.set_description(f"loss: {loss.item()} loss(avg): {np.mean(losses_train)}")

            running_loss /= self.max_num_steps
            losses_train_epoch.append(running_loss)

            self.writer.add_scalar("Training Loss | Epoch", running_loss, epoch)

            # Validation step
            self.model.eval()
            running_val_loss = 0.0

            with torch.no_grad():
                for idx, data in enumerate(val_loader):
                    if idx == self.max_num_steps_val:
                        break

                    val_loss, _ = self.model.forward_pass(data, self.device, self.criterion)

                    running_val_loss += val_loss.item()
                    
            running_val_loss /= self.max_num_steps_val
            losses_val_epoch.append(running_val_loss)

            self.scheduler.step(running_val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.writer.add_scalar("Learning Rate", current_lr, epoch)
            self.writer.add_scalar("Validation Loss", running_val_loss, epoch)

            print(f"Epoch {epoch + 1} | Train Loss: {running_loss:.4f} | Val Loss: {running_val_loss:.4f} | LR: {current_lr:.6f}")

        if checkpoint is not None:
            torch.save(self.model.state_dict(), os.path.join(self.exp_dir, f"epoch_{checkpoint}.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.exp_dir, f"epoch_0.pth"))
        
        self.writer.close()
        return losses_train, losses_val_epoch, losses_train_epoch

def train(self, checkpoint=None):
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
        
        if checkpoint is not None:
            torch.save(self.model.state_dict(), os.path.join(self.exp_dir, f"epoch_{checkpoint}.pth"))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.exp_dir, f"epoch_0.pth"))
        return losses_train
