from pathlib import Path
import logging
import urllib.parse
Log = logging.getLogger('rss')
import yaml

class Subscription:
	KNOWN_FIELDS = set('url base use_bayes enabled'.split())

	def __init__(self, key, url, base=None, use_bayes=True, enabled=True):
		Log.debug('Initializing subscription for URL: {0}'.format(url))
		self.key = key
		self.url = url
		self.base = set(base) if base else set()
		self.use_bayes = use_bayes
		self.enabled = enabled
	def __str__(self):
		return 'Subscription({0}={1})'.format(self.key, repr(self.url))
	def __repr__(self):
		return 'Subscription({0}={1}, {2})'.format(
				self.key, repr(self.url), ', '.join([
				'base={0}'.format(self.base),
				'use_bayes={0}'.format(self.use_bayes),
				'enabled={0}'.format(self.enabled),
					  ]))
	def get_mp_key(self):
		""" Key string value to group subscriptions by their
		network location (host), to prevent parallel simultaneous
		access to the same host which may trigger anti-DoS measures.
		"""
		parts = urllib.parse.urlparse(self.url)
		if parts.scheme == 'file':
			return self.url
		return parts.netloc

class Subscriptions:
	def __init__(self):
		self.subs = {}
	def load(self, *file_names):
		for file_name in map(Path, file_names):
			if not file_name.exists():
				Log.warning('Subscription file does not exist, skipping: {0}'.format(file_name))
				continue
			if file_name.suffix == '.ini':
				self.load_ini(file_name)
			elif file_name.suffix == '.yml':
				self.load_yaml(file_name)
			else:
				Log.error('Unknown type of subscription file: {0}'.format(file_name))
	def load_yaml(self, filename):
		try:
			data = yaml.safe_load(Path(filename).read_text())
			if not isinstance(data, dict):
				Log.error("{0}: Subscription file should be a YAML dict, instead got: {1}".format(filename, type(data)))
				return
			for key, definition in data.items():
				if definition is not None and not isinstance(data, dict):
					Log.error("{0}: {1}: Subscription item should be a dict, instead got: {2}".format(filename, key, type(definition)))
					continue
				self._load_single_sub(filename, key, definition)
		except Exception as e:
			Log.error("Failed to load YAML subscription file: {0}: {1}".format(filename, e))
	def _load_single_sub(self, filename, key, definition):
		try:
			sub = Subscription(key, None)
			if definition:
				for field in set(definition.keys()) - Subscription.KNOWN_FIELDS:
					Log.error("{0}: {1}: unknown field: {2}".format(filename, key, field))
				for field in Subscription.KNOWN_FIELDS:
					if field not in definition:
						continue
					setattr(sub, field, definition[field])
			if key in self.subs:
				Log.error("{0}: {1}: Duplicated subscription, ignoring redundant entry...".format(filename, key))
			else:
				self.subs[key] = sub
				return True
		except Exception as e:
			Log.error("{0}: Failed to parse subscription: {1}: {2}".format(filename, key, e))
		return False
	def load_ini(self, file_name):
		Log.debug('Parsing legacy INI file: {0}'.format(file_name))
		loaded_items = 0
		with open(str(file_name)) as f:
			current_group = ''
			for line in f.readlines():
				line = line.strip()
				if line.startswith('#') or not line:
					continue
				if line.startswith('['):
					group = line.lstrip('[').rstrip(']')
					Log.debug('  Group: {0}'.format(group))
					continue
				if group:
					Log.debug('    Line: {0}'.format(line))
					url = line
					use_bayes = True
					if url.startswith('+'):
						Log.debug('      Bayes is switched off.')
						url = url.lstrip('+')
						use_bayes = None

					key = '{0}_{1}'.format(group, url)
					if self._load_single_sub(file_name, key, {
						'url' : url,
						'base' : {group},
						'use_bayes' : use_bayes,
						}):
						loaded_items += 1
		Log.debug('Loaded {0} subscriptions.'.format(loaded_items))
	def iter(self, groups=None):
		groups = set(groups) if groups else set()
		Log.debug('Group filter: {0}'.format(groups))
		available_groups = set()
		for sub in self.subs.values():
			if sub.url is None:
				continue
			Log.debug('Processing subscription: {0}'.format(sub))
			available_groups |= sub.base
			if groups and not (sub.base & groups):
				Log.debug('  {0}: does not match groups: {1}'.format(sub.base, groups))
				continue
			if not sub.enabled:
				Log.debug('  Disabled.')
				continue
			Log.debug('  Ready to fetch.')
			yield sub
		for incorrect_group in groups - available_groups:
			Log.warning("Group '{0}' is not available in links!".format(incorrect_group))
