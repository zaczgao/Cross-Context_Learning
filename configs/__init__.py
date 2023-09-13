import importlib

from configs.config import BaseConfig


def find_cfg_using_name(args):
	cfg_filename = "configs." + args.model.lower() + "." + args.dataset + "cfg"
	cfglib = importlib.import_module(cfg_filename)
	cfg = None
	target_cfg_name = args.dataset + 'config'
	for name, cls in cfglib.__dict__.items():
		if name.lower() == target_cfg_name.lower() \
				and issubclass(cls, BaseConfig):
			cfg = cls

	if cfg is None:
		print("In %s.py, there should be a subclass of BaseConfig with class name that matches %s in lowercase." % (
		cfg_filename, target_cfg_name))
		exit(0)

	return cfg


def createConfig(args):
	"""Create a model given the option.

	This function warps the class CustomDatasetDataLoader.
	This is the main interface between this package and 'train.py'/'test.py'

	Example:
		>>> from models import create_model
		>>> model = create_model(opt)
	"""
	cfg = find_cfg_using_name(args)
	instance = cfg(args.phase, args.debug)
	print("config [%s] was created" % type(instance).__name__)

	assert instance.DATA["type"].lower() == args.dataset.lower()
	assert instance.MODEL["arch"].lower() == args.arch.lower()
	assert instance.MODEL["type"].lower() == args.model.lower()
	instance.NAME = args.name
	instance.display()

	return instance