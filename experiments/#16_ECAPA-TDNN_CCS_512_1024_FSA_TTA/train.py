import os
from this import d
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch

from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.cm as cm

class MGCTrainer:
    args = None
    model = None
    logger = None
    train_loader = None
    val_loader = None
    eval_loader = None
    genres = None
    optimizer = None
    lr_scheduler = None

    def run(self):
        self.name = self.args['name']
        self.best_acc = 0
        self.best_name = None
        
        for epoch in range(1, self.args['epoch'] + 1):
            # train
            self.train(epoch)
            
            # validation
            if epoch % 1 == 0:
                acc, plot = self.test(self.val_loader)
                self.logger.log_metric('val/accuracy', acc, epoch)
                if self.best_acc < acc:
                    self.best_acc = acc
                    self.best_state = self.model.state_dict()
                    self.logger.log_metric('val/best', acc, epoch)
                    if self.args['epoch'] * 0.2 < epoch:
                        self.logger.save_model(f'{self.name}_BestModel_ACC{acc:.4f}', self.model.state_dict())
                        self.logger.log_image(f'val_{acc:.4f}', plot)
            self.lr_scheduler.step()
        
        # evaluation
        print('load best Model')
        self.model.load_state_dict(
            self.best_state
        )
        self.model.eval()
        print('Evaluation start')
        acc, plot = self.test(self.eval_loader)
        self.logger.log_metric('eval/accuracy', acc, epoch)
        self.logger.log_image(f"evaluation_{acc:.4f}", plot)

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
        crop_size = self.args['crop_size']
        n = int(30 // (crop_size // 100)) + 1
        max_audio = int(crop_size * 220.5 + 256)

        correct = [0 for _ in range(10)]
        total = [0 for _ in range(10)]
        with torch.set_grad_enabled(False):
            for x, label in tqdm(loader, desc='test', ncols=90):                
                # to cuda
                x = x.numpy()
                x = x[0]

                # TTA
                feats = []
                startframe = np.linspace(0, x.shape[0]-max_audio, num=n)
                for asf in startframe:
                    feats.append(x[int(asf):int(asf)+max_audio])
                feats = np.stack(feats, axis = 0).astype(np.float)
                x = torch.FloatTensor(feats).cuda()

                # feed forward
                prediction = self.model(x)
                prediction = torch.sum(prediction, dim=0)

                # count
                label = label.item()
                prediction = torch.max(prediction, dim=0)[1].item()

                if prediction == label:
                    correct[label] += 1
                total[label] += 1

        c = 0 
        t = 0
        map = np.zeros(11)
        for i in range(10):
            c += correct[i]
            t += total[i]
            map[i] = correct[i] / total[i] * 100
        acc = c / t * 100
        map[10] = acc
        plot = self.get_plot(map)

        return acc, plot

    def get_plot(self, data):
        x = np.arange(11)
        plt.bar(x, data, color="skyblue", width=0.5)
        plt.xticks(x, self.genres + ['TOTAL'], rotation=90)
        
        plt.title("Test", fontsize=20)

        plt.xlabel("class", fontsize=14)
        plt.ylabel("accuracy", fontsize=14)
        plt.tick_params(axis='x', width=2)
        
        plt.savefig(self.name+"_plot.png", dpi=400, bbox_inches="tight")
        plt.close()
        img = Image.open(self.name+"_plot.png")
        os.remove(self.name+"_plot.png")

        return img
    
    def get_t_sne(self,title, dimension, num_class, x):
        genre, num, feats = 10, 10, 3072   #np.shape(x)

        x = np.reshape(x, (-1, feats))
        print(np.shape(x))
        x = TSNE(n_components=2).fit_transform(x)
        x = np.reshape(x, (genre, num, 2))

        colors = cm.rainbow(np.linspace(0, 1, num_class))
        for i in range(num_class):
            plt.scatter(x[i, :, 0], x[i, :, 1], color=colors[i])
            #plt.scatter(x[i, 1, :, 0], x[i, 1, :, 1], color=colors[i], marker='x')

        plt.title(title, fontsize=20)
        plt.savefig("t_sne.png", dpi=400, bbox_inches="tight")
        plt.close()
        img = Image.open("t_sne.png")
        os.remove('t_sne.png')

        return img
    
    def load_parameters(self, path):
        self_state = self.model.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)