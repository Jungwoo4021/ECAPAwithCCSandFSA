import os

from .interface import ExperimentLogger
from .local import LocalLogger
from .wandb import WandbLogger

class LogModuleController(ExperimentLogger):
	def __init__(self, modules):
		self.components = modules

	def log_metric(self, name, value, step=None):
		for component in self.components:
			component.log_metric(name, value, step)

	def log_text(self, name, text):
		for component in self.components:
			component.log_text(name, text)

	def log_parameter(self, dictionary):
		for component in self.components:
			component.log_parameter(dictionary)

	def log_image(self, name, image):
		for component in self.components:
			component.log_image(name, image)

	def save_model(self, name, model):
		for component in self.components:
			component.save_model(name, model)

	def finish(self):
		for component in self.components:
			component.finish()
	
	class Builder():
		def __init__(self, name, project):
			self.args = {}
			self.args['name'] = name
			self.args['project'] = project
			self.args['flag_use_local'] = False
			self.args['flag_use_wandb'] = False
			self.args['tags'] = []
			self.args['description'] = ''
			self.args['source_path'] = None
			self.args['source_files'] = None
		
		def tags(self, tags):
			if type(tags) is list:
				self.args['tags'] = tags
				return self
			else:
				raise Exception(f'tags must be list. Given type is {type(tags)}')

		def description(self, description):
			self.args['description'] = description
			return self

		def save_source_files(self, path):
			self.args['source_path'] = path
			self.args['source_files'] = []
			for p, _, fs in os.walk(path):
				for f in fs:
					ext = os.path.splitext(f)[-1]
					if ext == '.py' and 'vscode' not in p:
						self.args['source_files'].append(
							f"{p}/{f}")
			return self

		def use_wandb(self, entity, api_token):
			self.args['flag_use_wandb'] = True
			self.args['wandb_entity'] = entity
			self.args['wandb_api_token'] = api_token
			return self

		def use_local(self, path):
			self.args['flag_use_local'] = True
			self.args['local_path'] = path
			return self

		def build(self):
			modules = []

			if self.args['flag_use_wandb']:
				wandb = WandbLogger( 
					self.args['wandb_api_token'],
					self.args['source_path'], 
					self.args['name'], 
					self.args['project'], 
					self.args['wandb_entity'],
					self.args['tags']
				)
				modules.append(wandb)

			if self.args['flag_use_local']:
				local = LocalLogger(
					self.args['local_path'], 
					self.args['name'], 
					self.args['project'], 
					self.args['tags'], 
					self.args['description'],
					self.args['source_path']
				)
				modules.append(local)
			
			package = LogModuleController(modules)

			return package