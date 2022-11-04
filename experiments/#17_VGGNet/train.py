import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch

from tqdm import tqdm

class MGCTrainer:
    args = None
    model = None
    melon = None
    logger = None
    train_loader = None
    val_loader = None
    eval_loader = None
    genres = None
    optimizer = None
    lr_scheduler = None

    def run(self):
        self.best_acc = 0
        self.best_name = None

        for epoch in range(1, self.args['epoch'] + 1):
            # train
            self.train(epoch)
            
            # validation
            if epoch % 5 == 0:
                acc, plot = self.test(self.val_loader)
                self.logger.log_metric('val/accuracy', acc, epoch)
                if self.best_acc < acc:
                    self.best_acc = acc
                    self.best_state = self.model.state_dict()
                    self.logger.log_metric('val/best', acc, epoch)
                    if self.args['epoch'] * 0.2 < epoch:
                        self.logger.save_model(f'BestModel_ACC{acc:.4f}', self.model.state_dict())
                        self.logger.log_image(f'val_{acc:.4f}', plot)
            self.lr_scheduler.step()
            
        # evaluation
        self.model.load_state_dict(
            self.best_state
        )
        self.model.eval()
        acc, plot = self.test(self.eval_loader)
        self.logger.log_metric('eval/accuracy', acc, epoch)
        self.logger.log_image("evaluation", plot)

    def train(self, epoch):
        # set train mode
        self.model.train()
        
        count = 0
        loss_sum = 0
        with tqdm(total=len(self.train_loader), ncols=90) as pbar:
            for x, label in self.train_loader:
                # clear grad
                self.optimizer.zero_grad()

                # to cuda
                x = x.cuda()
                label = label.cuda()

                # feed forward
                loss = self.model(x, label)

                # backpropagation
                loss.backward()
                self.optimizer.step()

                # log
                count += 1
                loss_sum += loss.item()
                if 100 < count:
                    self.logger.log_metric('Loss', loss_sum / count)
                    count = 0
                    loss_sum = 0
                
                # pbar
                desc = f'{self.args["name"]}-[{epoch}/{self.args["epoch"]}]|(loss): {loss.item():.3f}'
                pbar.set_description(desc)
                pbar.update(1)
    
    def test(self, loader):
        # set test mode
        self.model.eval()

        correct = [0 for _ in range(28)]
        total = [0 for _ in range(28)]
        with torch.set_grad_enabled(False):
            for x, label in tqdm(loader, desc='test', ncols=90):
                _, num_seg, bins, times = x.size()

                # to cuda
                x = x.cuda().view(num_seg, bins, times)

                # feed forward
                p = self.model(x)

                # count
                label = label.item()
                p = p.mean(dim=0, keepdim=True)
                p = torch.max(p, dim=1)[1].item()
                
                if p == label:
                    correct[label] += 1
                total[label] += 1

        c = 0 
        t = 0
        map = np.zeros(29)
        for i in range(28):
            c += correct[i]
            t += total[i]
            map[i] = correct[i] / total[i] * 100
        acc = c / t * 100
        map[28] = acc
        plot = self.get_plot(map)

        return acc, plot

    def get_plot(self, data):
        x = np.arange(29)
        plt.bar(x, data, color="skyblue", width=0.5)
        plt.xticks(x, self.genres + ['TOTAL'], rotation=90)
        
        plt.title("Test", fontsize=20)

        plt.xlabel("class", fontsize=14)
        plt.ylabel("accuracy", fontsize=14)
        plt.tick_params(axis='x', width=2)
        
        plt.savefig("plot.png", dpi=400, bbox_inches="tight")
        plt.close()
        img = Image.open("plot.png")
        os.remove('plot.png')

        return img