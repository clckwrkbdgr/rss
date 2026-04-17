import logging
import urllib.parse
Log = logging.getLogger('rss')

class Subscription:
	def __init__(self, url, base=None, use_bayes=False):
		Log.debug('Initializing subscription for URL: {0}'.format(url))
		self.url = url
		self.base = set(base) if base else set()
		self.use_bayes = use_bayes
	def __repr__(self):
		return 'Subscription({0})'.format(repr(self.url))
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
		self.subs = []
	def load(self, file_name):
		result = []
		Log.debug('Parsing file: {0}'.format(file_name))
		with open(file_name) as f:
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
					result.append(Subscription(url, base={group}, use_bayes=use_bayes))
		Log.debug('Loaded {0} subscriptions.'.format(len(result)))
		self.subs = result
	def iter(self, groups=None):
		groups = set(groups) if groups else set()
		Log.debug('Group filter: {0}'.format(groups))
		available_groups = set()
		for sub in self.subs:
			Log.debug('Processing subscription: {0}'.format(sub))
			available_groups |= sub.base
			if groups and not (sub.base & groups):
				Log.debug('  {0}: does not match groups: {1}'.format(sub.base, groups))
				continue
			Log.debug('  Ready to fetch.')
			yield sub
		for incorrect_group in groups - available_groups:
			Log.warning("Group '{0}' is not available in links!".format(incorrect_group))
