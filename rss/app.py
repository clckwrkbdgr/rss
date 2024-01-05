import os

def get_data_dir():
	data_dir = os.environ.get('XDG_DATA_HOME')
	if not data_dir:
		data_dir = os.path.join(os.path.expanduser("~"), ".local", "share")
	rss_data_dir = os.path.join(data_dir, "rss")
	os.makedirs(rss_data_dir, exist_ok=True)
	return rss_data_dir

def get_cache_dir():
	cache_dir = os.environ.get('XDG_CACHE_HOME')
	if not cache_dir:
		cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
	rss_cache_dir = os.path.join(cache_dir, "rss")
	os.makedirs(rss_cache_dir, exist_ok=True)
	return rss_cache_dir

class Config:
	GUID_FILE = os.path.join(get_cache_dir(), "guids.sqlite")
	RSS_INI_FILE = os.path.join(get_data_dir(), 'rss.ini')
	RSS_DIR = os.path.join(os.path.expanduser("~"), 'RSS')
	TRAIN_ROOT_DIR = get_data_dir()

	def __init__(self, **kwargs):
		for name, value in kwargs.items():
			assert getattr(self, name)
			if value is not None:
				setattr(self, name, value)
