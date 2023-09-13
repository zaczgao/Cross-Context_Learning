import importlib


def find_transform_using_name(transform_name):
	"""Import the module "models/[model_name]_model.py".

	In the file, the class called DatasetNameModel() will
	be instantiated. It has to be a subclass of BaseModel,
	and it is case-insensitive.
	"""
	transform_filename = "samples.transforms.ssl_transforms"
	transformlib = importlib.import_module(transform_filename)
	transform = None
	target_transform_name = transform_name
	for name, cls in transformlib.__dict__.items():
		if name.lower() == target_transform_name.lower() \
				and issubclass(cls, object):
			transform = cls

	if transform is None:
		print("In %s.py, there should be a subclass of object with class name that matches %s in lowercase." % (
		transform_filename, target_transform_name))
		exit(0)

	return transform


def get_trasnform(isTrain, cfg):
	"""Create a model given the option.

	This function warps the class CustomDatasetDataLoader.
	This is the main interface between this package and 'train.py'/'test.py'

	Example:
		>>> from models import create_model
		>>> model = create_model(opt)
	"""
	cfg_transform = cfg.DATA["transform"].copy()
	obj_type = cfg_transform.pop('type')

	if cfg.PHASE == "pretrain":
		transform = find_transform_using_name(obj_type)
	else:
		transform = find_transform_using_name("lincls_transforms")

	instance = transform(isTrain, **cfg_transform)
	print("transform {:s} for phase {:s} was created".format(type(instance).__name__, cfg.PHASE))
	return instance
