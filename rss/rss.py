#!/usr/bin/python3.3
import bs4
import xml.etree.ElementTree as ET
import xml.parsers.expat
import os
import os.path
import logging
import sys
import socket, threading, multiprocessing.pool, signal
import difflib
import urllib.request, urllib.parse
import re
import pprint, textwrap
import datetime
import http
import random
import gzip
import logging
import resource, tracemalloc, gc
from collections import defaultdict
from . import guids
from . import wwts
from . import app
Log = logging.getLogger('rss')
log = Log.warning

def get_log_file(*args):
	data_dir = os.environ.get('XDG_STATE_HOME')
	if not data_dir:
		data_dir = os.path.join(os.path.expanduser("~"), ".state")
	logdir = os.path.join(data_dir, "rss")
	os.makedirs(logdir, exist_ok=True)
	return os.path.join(logdir, "rss.log")

UNPRINTABLE = r'/?'
MAX_FILE_NAME_LENGTH = 70
HTML_TEMPLATE = """<html>
<head>
<meta http-equiv=Content-Type content="text/html; charset=utf-8"/>
<style type="text/css">
	html body {{ background-color: #111 }}
	body {{ color: #bbb }}
	a {{ color:#b91 }}
</style>
<title>{0}</title>
</head>
<body>
<h1>{0}</h1>
<p><a href="{1}">{1}</a></p>
<p>{2}</p>
<div>{3}</div>
</body>
</html>"""
HOSTS = [
		"8.8.8.8", # Google DNS
		"8.8.4.4", # Google DNS
		"139.130.4.5", # Australian primary NS
		"208.67.222.222", # OpenDNS
		"208.67.220.220" # OpenDNS
		]

def knock(host):
	return os.system("ping -c 1 -s 1 -W 2 {0} >/dev/null 2>&1".format(host)) == 0

def check_network():
	host = random.choice(HOSTS)
	if knock(host):
		return True
	for host in HOSTS:
		if knock(host):
			return True
	return False

def isonow():
	return datetime.datetime.now().isoformat(' ')

def load_ini(file_name):
	result = {}
	with open(file_name) as f:
		current_group = ''
		for line in f.readlines():
			line = line.strip()
			if line.startswith('#') or not line:
				continue
			if line.startswith('['):
				group = line.lstrip('[').rstrip(']')
				result[group] = []
				continue
			if group:
				result[group].append(line)
	return result

def fetch_items(root, url=None):
	items = []
	if root.tag == 'rss':
		channel = root.find('channel')
		if channel is None:
			log('{1}: Malformed RSS structure, <channel> tag is missing: {0}'.format(ET.tostring(root, encoding='utf-8'), url))
			return items
		items = channel.findall('item')
	elif root.tag == '{http://www.w3.org/2005/Atom}feed':
		items = root.findall('{http://www.w3.org/2005/Atom}entry')
	return items

def get_guid(item):
	for tagname in ['guid', '{http://www.w3.org/2005/Atom}guid', 'id', '{http://www.w3.org/2005/Atom}id', 'link', '{http://www.w3.org/2005/Atom}link']:
		result = item.find(tagname)
		if result is not None:
			if result.tag.endswith('link') and 'href' in result.attrib:
				return result.attrib['href'].lower()
			if result.tag.endswith('guid') and 'isPermalink' in result.attrib:
				return result.text.lower()
			return result.text
	return ''

def get_title(item):
	for tagname in ['title', '{http://www.w3.org/2005/Atom}title', 'link', '{http://www.w3.org/2005/Atom}link']:
		result = item.find(tagname)
		if result is not None:
			text = result.attrib['href'] if (result.tag.endswith('link') and 'href' in result.attrib) else result.text
			if text is not None and text.count('"') % 2 != 0 and text.endswith('"'):
				text = text[:-1]
			return text or ''
	return ''

def get_date(item):
	for tagname in ['pubDate', '{http://www.w3.org/2005/Atom}pubDate', 'updated', '{http://www.w3.org/2005/Atom}updated']:
		result = item.find(tagname)
		if result is not None:
			return result.text
	return ''

def get_link(item):
	links = {}
	for tagname in ['link', '{http://www.w3.org/2005/Atom}link']:
		for result in item.findall(tagname):
			rel = result.attrib['rel'] if 'rel' in result.attrib else 'self'
			if 'href' in result.attrib:
				links[rel] = result.attrib['href']
			else:
				links[rel] = result.text
	if 'alternate' in links:
		return links['alternate']
	if 'self' in links:
		return links['self']
	if links:
		return links[links.keys()[0]]
	return ''

def get_content(item):
	result = None
	tags = ['fulltext', 'description', 'summary', 'content']
	fulltags = []
	for tag in tags:
		fulltags += [tag, '{http://www.w3.org/2005/Atom}' + tag]
	for tagname in fulltags:
		result = item.find(tagname)
		if result is not None:
			if result.text is None:
				return ET.tostring(result, encoding='utf-8').decode('utf-8')
			else:
				return result.text
	return ''

def interrupt_fetch(feed, handle):
	log('Feed download interrupted: {0}'.format(feed))
	handle.close()
	import _thread
	_thread.interrupt_main()

DOCTYPE = b'''
<!DOCTYPE naughtyxml [
	<!ENTITY nbsp "&#0160;">
	<!ENTITY copy "&#0169;">
	<!ENTITY laquo "&#0171;">
	<!ENTITY raquo "&#0187;">
	<!ENTITY ndash "&#8211;">
	<!ENTITY mdash "&#8211;">
	<!ENTITY bull "&#8226;">
]>
'''.replace(b'\n', b'')
# Yields: guid, title, date, link, content
def parse_feed(url, attempts_left=3):
	try:
		text = fetch_url(url, attempts_left=attempts_left)
		if text is None:
			return
		yield from parse_text(text, url)
	except KeyboardInterrupt:
		log('{0}: Feed download interrupted'.format(url))
	except Exception as e:
		Log.exception('Unknown exception {1} when parsing feed: {0}'.format(url, e))

def fetch_url(url, attempts_left=3):
	try:
		Log.debug('Requesting...')
		req = urllib.request.Request(url, headers={ 'User-Agent': 'Mozilla/5.0 (Linux)' })

		handle = urllib.request.urlopen(req, timeout=30)

		timer = threading.Timer(60, interrupt_fetch, (url, handle))
		timer.start()
		text = None
		try:
			text = handle.read()
		finally:
			timer.cancel()
		return text
	except http.client.IncompleteRead as e:
		if attempts_left > 0:
			return fetch_url(url, attempts_left - 1)
		else:
			log('{0}: incomplete read: {1}'.format(url, e))
	except http.client.BadStatusLine as e:
		log('{0}: bad status line: {1}'.format(url, e))
	except urllib.error.URLError as e:
		try:
			e = e.args[0]
			if isinstance(e, OSError) and e.errno == 110:
				if attempts_left > 0:
					return fetch_url(url, attempts_left - 1)
				else:
					log('{0}: url: {1}'.format(url, e))
			else:
				log('{0}: url: {1}'.format(url, e))
		except:
			log('{0}: url: {1}'.format(url, e))
	except socket.error as e:
		try:
			if e.code == 500:
				if attempts_left > 0:
					return fetch_url(url, attempts_left - 1)
				else:
					log('{1}: socket({0}): {2}'.format(e.code, url, e))
			else:
				log('{1}: socket({0}): {2}'.format(e.code, url, e))
		except:
			log('{0}: socket: {1}'.format(url, e))
	return None

def parse_text(text, url, attempts_left=3):
	try:
		if len(text) > 2 and text[0] == 0x1f and text[1] == 0x8b:
			Log.debug('We have gzipped content here.')
			text = gzip.decompress(text)
		text = text.lstrip()
		text = text.replace(b'\x0d', b' ')
		text = text.translate(None,
				delete=b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f'
				)
		text = text.replace(b'& ', b'&amp; ')
		text = re.sub(r'&([^;]{10})'.encode(), r'&amp;\1'.encode(), text)
		rss_end_tag = text.find(b'</rss>')
		if rss_end_tag > -1:
			text = text[:rss_end_tag+len(b'</rss>')]
		if attempts_left == 1:
			text = text.replace(b'\x92', b"'")
		if attempts_left == 0:
			pass #text = text.replace(b'\xfc', b'u') # Quickfix for incorrect encoding.
		Log.debug('Loaded raw data: {0}'.format(textwrap.shorten(repr(text[:120]), width=100)))
		root = ET.fromstring(text) # FIXME replace with SAX, mind about reversed() below.
		if root.tag not in ['rss', '{http://www.w3.org/2005/Atom}feed']:
			log('{0} at {1} instead of <rss> or <feed>'.format(root.tag, url))
			return
		# Normally RSS feed contains most recent items on top,
		# so we reverse the list to match order of items with expected (natural) order of processing,
		# e.g. so that mtimes of created files were placed in order the items were created.
		Log.debug('Parsing XML...')
		for item in reversed(fetch_items(root, url=url)):
			title = get_title(item)
			title = title.strip() or title
			Log.debug('Fetched item: {0}'.format(repr(title)))
			guid = get_guid(item)
			if guid is None:
				log('{0}: no guid element, skipping: {1}'.format(url, item))
				continue
			yield guid, title, get_date(item), get_link(item), get_content(item)
		del text # To force garbage collection.
		del root # To force garbage collection.
	except UnicodeEncodeError as e:
		log('{0}: unicode: {1}'.format(url, e))
	except xml.etree.ElementTree.ParseError as e:
		Log.debug(e)
		if attempts_left > 0:
			if 'undefined entity' in str(e):
				xml_decl_start = text.find(b'<') + 1
				xml_decl_end = text.find(b'>') + 1
				if text[xml_decl_start:xml_decl_start+4] != b'?xml':
					text = b'<?xml version="1.0" encoding="UTF-8"?>' + DOCTYPE + text
				else:
					text = text[:xml_decl_end] + DOCTYPE + text[xml_decl_end:]
			yield from parse_text(text, url, attempts_left - 1)
		else:
			incomplete_read_patterns = [
					"no element found: line 6, column 0",
					"no element found: line 7",
					"unclosed CDATA section: line 7",
					"unclosed token: line 7",
					]
			if not any(pattern in str(e) for pattern in incomplete_read_patterns):
				log('{0}: parse: {1}'.format(url, e))
	except xml.parsers.expat.ExpatError as e:
		log('{0}: expat: {1}'.format(url, e))

def make_text(title, date, link, content):
	return HTML_TEMPLATE.format(title, link, date, content)

def extract_tags_from_text(text):
	try:
		import warnings
		with warnings.catch_warnings():
			# Ignore MarkupResemblesLocatorWarning
			warnings.simplefilter('ignore')
			soup = bs4.BeautifulSoup(text)
		tags = soup.find_all('a', class_='tag')
		return [tag.text for tag in tags]
	except TypeError as e:
		log('type: {0}'.format(e))
	return []

def make_filename(path, title, text):
	if not os.path.exists(path):
		os.makedirs(path)
	filename = title if title else isonow()
	filename = ''.join(['[{0}]'.format(tag) for tag in extract_tags_from_text(text)]) + filename
	filename = filename.replace('/', '_')
	filename = filename.replace('\\', '_')
	filename = filename.replace('\n', '_')
	filename = filename[:MAX_FILE_NAME_LENGTH]
	while os.path.exists(os.path.join(path, filename + '.html')):
		filename += '_'
	return os.path.join(path, filename + '.html')

pull_feed_lock = threading.Lock()

def pull_feed(config, group, url, db, use_bayes):
	if use_bayes:
		Log.debug('Opening Bayes from dir: {0}'.format(config.TRAIN_ROOT_DIR))
		bayes = wwts.Bayes(config, tokenizer=wwts.Tokenizer(lower=True))
		try:
			bayes.load()
			Log.debug('Loaded bayes data.')
		except Exception as e:
			log('bayes: {0}'.format(e))
	else:
		bayes = None
	for guid, title, date, link, content in parse_feed(url):
		with pull_feed_lock:
			if db.guid_exists(url, guid):
				Log.debug('GUID already exists, skipping.')
				continue
			if guid.startswith('http://') and db.guid_exists(url, guid.replace('http://', 'https://')):
				Log.debug('GUID already exists (http<->https), skipping.')
				continue
			if guid.startswith('https://') and db.guid_exists(url, guid.replace('https://', 'http://')):
				Log.debug('GUID already exists (https<->http), skipping.')
				continue

		savedir = config.RSS_DIR
		if bayes is not None:
			Log.debug('Guessing Bayes tag...')
			text_to_guess = content if content is not None else ""
			text_to_guess += title if title is not None else ""
			text_to_guess += link if link is not None else ""
			bayes_result = dict(bayes.guess(text_to_guess if text_to_guess else url))
			if 'good' not in bayes_result:
				bayes_result['good'] = 0
			if 'bad' not in bayes_result:
				bayes_result['bad'] = 0
			if bayes_result['bad'] > bayes_result['good']:
				savedir = os.path.join(savedir, 'unwanted')
				Log.debug('  Bad. Saving to: {0}'.format(savedir))
			else:
				savedir = os.path.join(savedir, group)
				Log.debug('  Good. Saving to: {0}'.format(savedir))
		else:
			Log.debug('  Bayes is off. Assuming as always good.')
			savedir = os.path.join(savedir, group)
			Log.debug('  Saving to: {0}'.format(savedir))

		text = make_text(title, date, link, content)
		if 'twitter.com' in url or 'twitter-rss.com' in url:
			parts = url.split('/');
			title = url[-1] + '_' + title
		with pull_feed_lock:
			filename = make_filename(savedir, title, content)
			Log.debug('Saving as: {0}'.format(filename))
			if not os.path.exists(os.path.dirname(filename)):
				os.makedirs(os.path.dirname(filename))
		with open(filename, 'w') as f:
			f.write(text)
		Log.debug('Remembering GUID: {0}'.format(guid))
		with pull_feed_lock:
			db.add_guid(url, guid)

import click

def init_logger(logger, filename, debug=False):
	logger = logging.getLogger(logger)

	if logger.handlers:
		for _handler in logger.handlers[:]:
			logger.removeHandler(_handler)
	logger.propagate = False

	level = logging.WARNING
	if debug:
		level = logging.DEBUG

	if debug:
		stream_handler = logging.StreamHandler(sys.stderr)
		fmt_string = '[%(levelname)s] %(name)s: %(message)s'
		stream_handler.setFormatter(logging.Formatter(fmt_string,
			datefmt='%Y-%m-%d %H:%M:%S',
			))
		logger.addHandler(stream_handler)
	elif filename:
		file_handler = logging.FileHandler(str(filename), delay=True, encoding='utf-8')
		file_handler.setFormatter(logging.Formatter(
			'%(asctime)s:%(name)s:%(levelname)s: %(message)s',
			datefmt='%Y-%m-%d %H:%M:%S',
			))
		logger.addHandler(file_handler)

	logger.setLevel(level)

@click.command()
@click.option('--debug', is_flag=True, help='Print debug traces.')
@click.option('--test', help='Test single feed (URL or local path) and exit.')
@click.option('--guid-file', help='GUID file. Default is {0}.'.format(app.Config.GUID_FILE))
@click.option('--dest-dir', help='Directory to store downloaded feeds. Default is {0}.'.format(app.Config.RSS_DIR))
@click.option('--config-file', help='File with feed definitions. Default is {0}.'.format(app.Config.RSS_INI_FILE))
@click.option('--train-dir', help='Root directory for WWTS train files. Default is {0}.'.format(app.Config.TRAIN_ROOT_DIR))
@click.option('--threads', type=int, default=4, help='Enables fetching feeds in parallel threads with specified number of thread pool workers (default is 4). Set to 0 to disable thread pool.')
@click.argument('groups', nargs=-1)
def main(groups, debug=False, test=None,
	guid_file=None, dest_dir=None, config_file=None, train_dir=None,
		 threads=4,
	):
	""" Fetches given groups of feeds defined in RSS config file,
	parses and stores posts in dest. directory.
	GUID file is used to track already fetched feed items.
	"""
	assert threads >= 0 # TODO click type=... checker instead.

	init_logger('rss', get_log_file() if test is None else None, debug=debug)
	Log.debug('GUID file: {0}'.format(guid_file))
	Log.debug('RSS dir: {0}'.format(dest_dir))
	Log.debug('INI file: {0}'.format(config_file))
	Log.debug('WWTS train dir: {0}'.format(train_dir))
	config = app.Config(
		GUID_FILE=guid_file,
		RSS_DIR=dest_dir,
		RSS_INI_FILE=config_file,
		TRAIN_ROOT_DIR=train_dir,
		)

	if test:
		url = test
		if os.path.exists(url):
			url = 'file://' + url
		Log.debug('Fetching single feed: {0}'.format(url))
		for item in parse_feed(url):
			pprint.pprint(item)
		return

	if not check_network():
		log("Network is down")
		return

	Log.debug('Loading config file: {0}'.format(config.RSS_INI_FILE))
	rsslinks = load_ini(config.RSS_INI_FILE)
	Log.debug('Loaded {0} groups.'.format(len(rsslinks)))
	available_groups = rsslinks.keys()
	if not groups:
		Log.info('Groups were not specified. Fetching everything.')
		groups = available_groups
	has_incorrect_groups = False
	for group in groups:
		if group not in available_groups:
			log("Group '{0}' is not available in links!".format(group))
			has_incorrect_groups = True
	if has_incorrect_groups:
		log("Available groups: {0}".format(', '.join(["'{0}'".format(group) for group in available_groups])))
		groups = [group for group in groups if group in available_groups]
		log("Will load following groups: {0}".format(' '.join(groups)))

	Log.debug('Opening GUID file: {0}'.format(config.GUID_FILE))
	db = guids.GuidDatabase(config.GUID_FILE)

	jobs = defaultdict(list)
	for group in groups:
		Log.debug('Processing group: {0}'.format(group))
		for url in rsslinks[group]:
			Log.debug('Processing URL: {0}'.format(url))

			use_bayes = True
			if url.startswith('+'):
				Log.debug('  Bayes is switched off.')
				url = url.lstrip('+')
				use_bayes = None

			parts = urllib.parse.urlparse(url)
			if parts.scheme == 'file':
				job_key = url
			else:
				job_key = parts.netloc
			jobs[job_key].append((pull_feed, (config, group, url, db, use_bayes)))

	tracemalloc.start()
	Log.debug('Initial memory usage: maxrss={0} alloc={1}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, tracemalloc.get_traced_memory()))
	def _worker(job_key, job_group):
		Log.debug('Running jobs for {0}'.format(job_key))
		for job, args in job_group:
			Log.debug('Running job: {0}'.format(args))
			job(*args)
			Log.debug('Memory usage: maxrss={0} alloc={1} @ {2}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, tracemalloc.get_traced_memory(), args))
		gc.collect()
	if threads:
		with multiprocessing.pool.ThreadPool(processes=threads) as pool:
			list(pool.starmap(_worker, jobs.items()))
	else:
		for data in jobs.items():
			_worker(*data)
	Log.debug('Final memory usage: maxrss={0} alloc={1}'.format(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, tracemalloc.get_traced_memory()))
	tracemalloc.stop()

	db.close()

if __name__ == "__main__":
	main()
