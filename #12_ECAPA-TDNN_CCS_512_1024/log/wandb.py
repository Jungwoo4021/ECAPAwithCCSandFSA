import os
import torch
from .interface import ExperimentLogger

class WandbLogger(ExperimentLogger):
	"""Save experiment logs to wandb
	"""
	def __init__(self, wandb_api_token, path, name, project, entity, tags, save_dir = None):
		import wandb
		self.wandb = wandb
		os.system(f'wandb login {wandb_api_token}')
		self.run = self.wandb.init(
				project=project,
				entity=entity,
				tags=tags
			)
		self.wandb.run.name = name
		path = os.path.join("/", path, project, "/".join(tags), name)
		self.paths = {
			'model' : f'{path}/model',
		}
		# upload zip file
		#wandb.save(save_dir + "/script/script.zip")

	def log_metric(self, name, value, step=None):
		self.wandb.log({name: value, 'epoch': step})   

	def log_text(self, name, text):
		pass

	def log_image(self, name, image):
		self.wandb.log({name: [self.wandb.Image(image)]})

	def log_parameter(self, dictionary):
		self.wandb.config.update(dictionary)

	def save_model(self, name, state_dict):
		path = f'{self.paths["model"]}/{name}.pt'
		dirname = os.path.dirname(path)
		os.makedirs(dirname, exist_ok=True)
		torch.save(state_dict, path)

	def finish(self):
		self.wandb.finish()