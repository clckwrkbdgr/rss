#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys
import argparse
import traceback
import re
import codecs
from functools import reduce
import pickle
import sqlite3
from collections import defaultdict
#import lib.bayes as bayes

## BAYES START

# This module is part of the Divmod project and is Copyright 2003 Amir Bakhtiar:
# amir@divmod.org.  This is free software; you can redistribute it and/or
# modify it under the terms of version 2.1 of the GNU Lesser General Public
# License as published by the Free Software Foundation.
#

import os
import operator
try:
	import re2 as re
except ImportError:
	import re
import math

try:
	from sets import Set
except ImportError:
	Set = set
from . import app

class BayesData(dict):

	def __init__(self, name=''):
		self.name = name

class ProbCache:
	def __init__(self, dataClass=None):
		self.data = None
		if dataClass is None:
			self.dataClass = BayesData
		else:
			self.dataClass = dataClass
	def lock(self): pass
	def unlock(self): pass
	def invalidate(self):
		self.data = None
	def valid(self):
		return self.data is not None
	def clear(self):
		self.data = {}
	def set_value(self, pname, word, value):
		cacheDict = self.data.setdefault(pname, self.dataClass(pname))
		cacheDict[word] = value
	def getNames(self):
		return set(self.data.keys())
	def getProbs(self, pname, words):
		""" extracts the probabilities of tokens in a message
		"""
		pool = self.data[pname]
		probs = [(word, pool[word]) for word in words if word in pool]
		probs.sort(key=lambda x: (x[1], x[0]), reverse=True)
		return probs[:2048]

class ChainDelegate:
	class MultiDelegate:
		def __init__(self, name, instances):
			self._name = name
			self._instances = instances
		def __call__(self, *args, **kwargs):
			results = []
			str_results = defaultdict(list)
			for instance in self._instances:
				try:
					result = getattr(instance, self._name)(*args, **kwargs)
					results.append(result)
					str_results[str(result)].append(instance)
				except:
					import traceback
					str_results[traceback.format_exc()].append(instance)
			if len(str_results) > 1:
				print('Results are different: {0}( {1}, {2} )\n\t'.format(self._name, args, kwargs) + '\n\t'.join(list(str_results.keys())))
			return results[0] if results else None
	def __init__(self, instances):
		self._chain = instances
	def __getattr__(self, name):
		return self.MultiDelegate(name, self._chain)

class SQLProbCache:
	def __init__(self, dataClass=None): # FIXME all access should be via transactions
		self.in_transaction = False
		filename = os.path.join(app.get_cache_dir(), "prob.sqlite")
		self.conn = sqlite3.connect(filename, timeout=10, check_same_thread=False) # To allow multithreading access.
		self.conn.text_factory = str # To prevent some dummy encoding bug.
		self.conn.execute("""
				CREATE TABLE IF NOT EXISTS Probs (
					pool_name TEXT,
					word TEXT,
					value REAL,
					PRIMARY KEY (pool_name, word)
				)
				;""")
		self.conn.commit()
	def __del__(self): # TODO should be done explicitly, e.g. with help of context manager.
		self.conn.close()
	def lock(self):
		if self.in_transaction:
			return
		self.in_transaction = True
		self.conn.execute("""BEGIN;""")
	def unlock(self):
		if not self.in_transaction:
			return
		self.in_transaction = False
		self.conn.commit()
	def _commit(self):
		if self.in_transaction:
			return
		self.conn.commit()

	def invalidate(self):
		if not self.valid():
			return
		self.clear()
	def valid(self):
		c = self.conn.cursor()
		c.execute("""SELECT COUNT(*) FROM Probs;""")
		self.conn.commit()
		count = [int(f) for f, in c]
		return bool(count[0]) if count else False
	def clear(self):
		self.conn.execute("""DELETE FROM Probs;""")
		self._commit()
	def set_value(self, pname, word, value):
		self.conn.execute("""
				 INSERT INTO Probs(pool_name, word, value) VALUES (?, ?, ?)
				 ON CONFLICT(pool_name, word) DO UPDATE SET
				 value = excluded.value
				 ;""", (pname, word, value))
		self._commit()
	def getNames(self):
		c = self.conn.cursor()
		c.execute("""SELECT DISTINCT pool_name FROM Probs;""")
		self.conn.commit()
		return {f for f, in c}
	def getProbs(self, pname, words):
		""" extracts the probabilities of tokens in a message
		"""
		c = self.conn.cursor()
		c.execute("""
				SELECT word, value FROM Probs
				WHERE pool_name = ?
				AND word IN (""" + ','.join(['?']*len(words)) + """)
				ORDER BY value DESC, word DESC
				LIMIT 2048
				;""", (pname,) + tuple(words))
		self.conn.commit()
		probs = [(word, value) for word, value in c]
		return probs

class SQLPools:
	def __init__(self, store_dir, dataClass=None):
		self.dataClass = dataClass
		self.store_dir = store_dir

		self.in_transaction = False
		self.conn = None
		self.cache = SQLProbCache(dataClass=self.dataClass)
	def __del__(self): # TODO should be done explicitly, e.g. with help of context manager.
		self.conn.close()
	def lock(self):
		if self.in_transaction:
			return
		self.in_transaction = True
		self.conn.execute("""BEGIN;""")
	def unlock(self):
		if not self.in_transaction:
			return
		self.in_transaction = False
		self.conn.commit()
	def _commit(self):
		if self.in_transaction:
			return
		self.conn.commit()

	def import_from(self, other, tracer=None):
		tracer = tracer or (lambda _:None)
		for name in self.poolNames():
			tracer('Removing pool {0}...'.format(repr(name)))
			self.removePool(name)
		for name in other.poolNames():
			tracer('Recreating pool {0}...'.format(repr(name)))
			self.newPool(name)
		for name in self.poolNames() + ['__Corpus__']:
			tracer('Repopulating pool {0}...'.format(repr(name)))
			for index, (token, count) in enumerate(other.iter_pool_words(name)):
				tracer('Repopulating pool {0} ({2}): {1}'.format(name, repr(token), index))
				self.conn.execute("""
					 INSERT INTO Tokens(pool_id, token, count)
					 VALUES ( (SELECT id FROM Pools WHERE name = ?), ?, ?)
					 ;""", (name, token, count))
		tracer('Commiting changes...')
		self._commit()
		tracer('Invalidating cache...')
		self.cache.invalidate()
		tracer('Imported successfully.')

	def get_pool_tokenCount(self, name):
		c = self.conn.cursor()
		c.execute("""
				SELECT SUM(count) FROM Pools, Tokens
				WHERE Pools.id = Tokens.pool_id
				AND Pools.name = ?
				;""", (name,))
		self.conn.commit()
		count = [int(f) for f, in c]
		return count[0] if count else 0
	def iter_pool_words(self, pname):
		c = self.conn.cursor()
		c.execute("""
				SELECT token, count
				FROM Tokens
				WHERE pool_id = (SELECT id FROM Pools WHERE name = ?)
				ORDER BY 1
				;""", (pname,))
		self.conn.commit()
		for word, thisCount in c:
			yield word, thisCount
	def iter_pool_corpus_words(self, pname):
		c = self.conn.cursor()
		c.execute("""
				SELECT corpus.token, corpus.count, this.count
				FROM Tokens corpus, Tokens this
				WHERE corpus.pool_id = (SELECT id FROM Pools WHERE name = '__Corpus__')
				AND this.pool_id = (SELECT id FROM Pools WHERE name = ?)
				AND corpus.token = this.token
				;""", (pname,))
		self.conn.commit()
		for word, totCount, thisCount in c:
			yield word, totCount, thisCount
	def pool_inc_token(self, pool_name, token):
		self.conn.execute("""
				 INSERT INTO Tokens(pool_id, token, count)
				 VALUES ( (SELECT id FROM Pools WHERE name = ?), ?, 1)
				 ON CONFLICT(pool_id, token) DO UPDATE SET
				 count = count + 1
				 ;""", (pool_name, token))
		self._commit()
		self.cache.invalidate()
	def pool_dec_token(self, pool_name, token):
		self.conn.execute("""
				 UPDATE Tokens
				 SET count = count - 1
				 WHERE pool_id = (SELECT id FROM Pools WHERE name = ?)
				 AND token = ?
				 ;""", (pool_name, token))
		self._commit()
		self.cache.invalidate()
	def has_pool(self, name):
		c = self.conn.cursor()
		c.execute("""
				SELECT COUNT(*) FROM Pools
				WHERE name = ?
				;""", (name,))
		self.conn.commit()
		count = [int(f) for f, in c]
		return bool(count[0]) if count else False

	def newPool(self, poolName):
		"""Create a new pool, without actually doing any
		training.
		"""
		if self.has_pool(poolName):
			return
		self.conn.execute("""
				 INSERT INTO Pools(name) VALUES (?)
				 ;""", (poolName,))
		self._commit()
		self.cache.invalidate() # not always true, but it's simple
	def removePool(self, poolName):
		self.conn.execute("""
				 DELETE FROM Tokens WHERE pool_id = (
					 SELECT id FROM Pools WHERE name = ?
				 )
				 ;""", (poolName,))
		self.conn.execute("""
				 DELETE FROM Pools WHERE name = ?
				 ;""", (poolName,))
		self._commit()
		self.cache.invalidate()
	def renamePool(self, poolName, newName):
		self.conn.execute("""
				 UPDATE Pools SET name = ? WHERE name = ?
				 ;""", (newName, poolName))
		self._commit()
		self.cache.invalidate()
	def mergePools(self, destPool, sourcePool):
		"""Merge an existing pool into another.
		The data from sourcePool is merged into destPool.
		The arguments are the names of the pools to be merged.
		The pool named sourcePool is left in tact and you may
		want to call removePool() to get rid of it.
		"""
		self.conn.execute("""
				 UPDATE Tokens destPool
				 SET count = count + (
					SELECT count FROM Tokens sourcePool, Pools
					WHERE sourcePool.pool_id = Pools.id
					AND Pools.name = ?
					AND sourcePool.token = destPool.token
					AND sourcePool.pool_id = destPool.pool_id
				 )
				 WHERE id = (SELECT id FROM Pools WHERE name = ?)
				 ;""", (sourcePool, destPool))
		self.conn.execute("""
				 INSERT INTO Tokens(pool_id, token, count)
				 SELECT destPool.pool_id, sourcePool.token, sourcePool.count
				 FROM Tokens sourcePool LEFT JOIN Tokens destPool
					 ON destPool.token = sourcePool.token
					 AND sourcePool.pool_id = (SELECT id FROM Pools WHERE name = ?)
					 AND destPool.pool_id = (SELECT id FROM Pools WHERE name = ?)
				 WHERE destPool.token IS NULL
				 ;""", (sourcePool, destPool))
		self._commit()
		self.cache.invalidate()
	def save(self, fname='train.sqlite'):
		pass # Saved automatically on commits.
	def load(self, fname='train.sqlite'):
		if '/' not in fname:
			fname = os.path.join(self.store_dir, fname)

		if self.conn:
			self.conn.close()

		self.conn = sqlite3.connect(fname, timeout=10, check_same_thread=False) # To allow multithreading access.
		self.conn.text_factory = str # To prevent some dummy encoding bug.
		self.conn.execute("""
				CREATE TABLE IF NOT EXISTS Pools (
					id INTEGER PRIMARY KEY AUTOINCREMENT,
					name TEXT
				)
				;""")
		self.conn.execute("""
				CREATE TABLE IF NOT EXISTS Tokens (
					pool_id INTEGER,
					token TEXT,
					count INTEGER,
					PRIMARY KEY (pool_id, token)
				)
				;""")
		self.conn.execute("""
				CREATE TRIGGER IF NOT EXISTS delete_tokens_with_zeroed_count
				AFTER UPDATE ON Tokens
				BEGIN
					DELETE FROM Tokens
					WHERE pool_id = NEW.pool_id AND token = NEW.token
					AND NEW.count <= 0;
				END
				;""")
		self.conn.commit()

		if not self.has_pool('__Corpus__'):
			self.newPool('__Corpus__')
		# Should invalidate cache in general case,
		# but let's hope it's called just once for a Bayes object,
		# so cache validity will be addressed later anyway:
		# re-create for pickle-based cache, re-use for SQL-based cache.
		#self.cache.invalidate()
	def poolNames(self):
		"""Return a sorted list of Pool names.
		Does not include the system pool '__Corpus__'.
		"""
		c = self.conn.cursor()
		c.execute("""
				SELECT name FROM Pools
				WHERE name <> '__Corpus__'
				;""")
		self.conn.commit()
		return sorted([f for f, in c])

	def _buildCache(self):
		""" merges corpora and computes probabilities
		"""
		if self.cache.valid():
			return
		self.cache.lock()
		self.cache.clear()
		self.lock()
		for pname in self.poolNames():
			poolCount = self.get_pool_tokenCount(pname)
			themCount = max(self.get_pool_tokenCount('__Corpus__') - poolCount, 1)
			for word, totCount, thisCount in self.iter_pool_corpus_words(pname):
				otherCount = float(totCount) - thisCount

				if not poolCount:
					goodMetric = 1.0
				else:
					goodMetric = min(1.0, otherCount/poolCount)
				badMetric = min(1.0, thisCount/themCount)
				if (goodMetric + badMetric) == 0:
					Log.error('Pool token has zero count: {0}: {1}'.format(pname, (word, totCount, thisCount)))
					continue
				f = badMetric / (goodMetric + badMetric)

				# PROBABILITY_THRESHOLD
				if abs(f-0.5) >= 0.1 :
					# GOOD_PROB, BAD_PROB
					self.cache.set_value(pname, word, max(0.0001, min(0.9999, f)))
		self.unlock()
		self.cache.unlock()
	def guessProbs(self, tokens, combiner):
		self._buildCache()
		res = {}
		for pname in self.cache.getNames():
			p = self.cache.getProbs(pname, tokens)
			if p:
				res[pname] = combiner(p, pname)
		res = list(res.items())
		res.sort(key=lambda x: x[1], reverse=True)
		return res

class Pools:
	def __init__(self, store_dir, dataClass=None):
		if dataClass is None:
			self.dataClass = BayesData
		else:
			self.dataClass = dataClass
		self.store_dir = store_dir
		self.pools = {}
		self.corpus = self.dataClass('__Corpus__')
		self.pools['__Corpus__'] = self.corpus
		self.cache = SQLProbCache(dataClass=self.dataClass)
	def lock(self): pass
	def unlock(self): pass
	def get_pool_tokenCount(self, name):
		pool = self.pools.get(name)
		return sum(pool.values())
	def iter_pool_words(self, pname):
		for word, count in sorted(self.pools[pname].items()):
			yield word, count
	def iter_pool_corpus_words(self, pname):
		for word, totCount in self.pools['__Corpus__'].items():
			# for every word in the copus
			# check to see if this pool contains this word
			thisCount = float(self.pools[pname].get(word, 0.0))
			if (thisCount == 0.0): continue
			yield word, totCount, thisCount
	def pool_inc_token(self, pool_name, token):
		pool = self.pools[pool_name]
		count = pool.get(token, 0)
		pool[token] =  count + 1
		self.cache.invalidate()
	def pool_dec_token(self, pool_name, token):
		pool = self.pools[pool_name]
		count = pool.get(token, 0)
		if count:
			if count == 1:
				del(pool[token])
			else:
				pool[token] =  count - 1
		self.cache.invalidate()
	def has_pool(self, name):
		return name in self.pools

	def newPool(self, poolName):
		"""Create a new pool, without actually doing any
		training.
		"""
		self.cache.invalidate() # not always true, but it's simple
		self.pools.setdefault(poolName, self.dataClass(poolName))
	def removePool(self, poolName):
		del(self.pools[poolName])
		self.cache.invalidate()
	def renamePool(self, poolName, newName):
		self.pools[newName] = self.pools[poolName]
		self.pools[newName].name = newName
		self.removePool(poolName)
		self.cache.invalidate()
	def mergePools(self, destPool, sourcePool):
		"""Merge an existing pool into another.
		The data from sourcePool is merged into destPool.
		The arguments are the names of the pools to be merged.
		The pool named sourcePool is left in tact and you may
		want to call removePool() to get rid of it.
		"""
		sp = self.pools[sourcePool]
		dp = self.pools[destPool]
		for tok, count in sp.items():
			if dp.get(tok):
				dp[tok] += count
			else:
				dp[tok] = count
		self.cache.invalidate()
	def save(self, fname='train.pkl'):
		if '/' not in fname:
			fname = os.path.join(self.store_dir, fname)
		fp = open(fname, 'wb')
		pickle.dump(self.pools, fp)
		fp.close()
	def load(self, fname='train.pkl'):
		if '/' not in fname:
			fname = os.path.join(self.store_dir, fname)
		fp = open(fname, 'rb')
		self.pools = pickle.load(fp)
		fp.close()
		# Should invalidate cache in general case,
		# but let's hope it's called just once for a Bayes object,
		# so cache validity will be addressed later anyway:
		# re-create for pickle-based cache, re-use for SQL-based cache.
		#self.cache.invalidate()
	def poolNames(self):
		"""Return a sorted list of Pool names.
		Does not include the system pool '__Corpus__'.
		"""
		pools = list(self.pools.keys())
		pools.remove('__Corpus__')
		pools = [pool for pool in pools]
		pools.sort()
		return pools

	def _buildCache(self):
		""" merges corpora and computes probabilities
		"""
		if self.cache.valid():
			return
		self.cache.lock()
		self.cache.clear()
		for pname in self.poolNames():
			poolCount = self.get_pool_tokenCount(pname)
			themCount = max(self.get_pool_tokenCount('__Corpus__') - poolCount, 1)
			for word, totCount, thisCount in self.iter_pool_corpus_words(pname):
				otherCount = float(totCount) - thisCount

				if not poolCount:
					goodMetric = 1.0
				else:
					goodMetric = min(1.0, otherCount/poolCount)
				badMetric = min(1.0, thisCount/themCount)
				f = badMetric / (goodMetric + badMetric)

				# PROBABILITY_THRESHOLD
				if abs(f-0.5) >= 0.1 :
					# GOOD_PROB, BAD_PROB
					self.cache.set_value(pname, word, max(0.0001, min(0.9999, f)))
		self.cache.unlock()
	def guessProbs(self, tokens, combiner):
		self._buildCache()
		res = {}
		for pname in self.cache.getNames():
			p = self.cache.getProbs(pname, tokens)
			if p:
				res[pname] = combiner(p, pname)
		res = list(res.items())
		res.sort(key=lambda x: x[1], reverse=True)
		return res

class Bayes(object):

	def __init__(self, config, tokenizer=None, combiner=None, dataClass=None):
		self.config = config
		if dataClass is None:
			self.dataClass = BayesData
		else:
			self.dataClass = dataClass
		self.pools = SQLPools(self.config.TRAIN_ROOT_DIR, dataClass=dataClass)
		# The tokenizer takes an object and returns
		# a list of strings
		if tokenizer is None:
			#self._tokenizer = Tokenizer()
			self._tokenizer = NGrams()
		else:
			self._tokenizer = tokenizer
		# The combiner combines probabilities
		if combiner is None:
			self.combiner = self.robinson
		else:
			self.combiner = combiner

	def commit(self):
		self.save()

	def newPool(self, poolName):
		self.pools.lock()
		self.pools.newPool(poolName)
		self.pools.unlock()

	def removePool(self, poolName):
		self.pools.lock()
		self.pools.removePool(poolName)
		self.pools.unlock()

	def renamePool(self, poolName, newName):
		self.pools.lock()
		self.pools.renamePool( poolName, newName)
		self.pools.unlock()

	def mergePools(self, destPool, sourcePool):
		self.pools.lock()
		self.pools.mergePools(destPool, sourcePool)
		self.pools.unlock()

	def save(self, fname='train.pkl'):
		self.pools.save()

	def load(self, fname='train.pkl'):
		self.pools.load()

	def poolNames(self):
		return self.pools.poolNames()

	def getTokens(self, obj):
		"""By default, we expect obj to be a screen and split
		it on whitespace.

		Note that this does not change the case.
		In some applications you may want to lowecase everthing
		so that "king" and "King" generate the same token.

		Override this in your subclass for objects other
		than text.

		Alternatively, you can pass in a tokenizer as part of
		instance creation.
		"""
		return self._tokenizer.tokenize(obj)

	def train(self, pool_name, item):
		"""Train Bayes by telling him that item belongs
		in pool. uid is optional and may be used to uniquely
		identify the item that is being trained on.
		"""
		tokens = self.getTokens(item)
		self.pools.newPool(pool_name)
		self._train(tokens, pool_name)

	def untrain(self, pool_name, item):
		tokens = self.getTokens(item)
		if not self.pools.has_pool(pool_name):
			return
		self._untrain(tokens, pool_name)

	def _train(self, tokens, pool_name):
		wc = 0
		self.pools.lock()
		for token in tokens:
			self.pools.pool_inc_token(pool_name, token)
			self.pools.pool_inc_token('__Corpus__', token)
			wc += 1
		self.pools.unlock()

	def _untrain(self, tokens, pool_name):
		self.pools.lock()
		for token in tokens:
			self.pools.pool_dec_token(pool_name, token)
			self.pools.pool_dec_token('__Corpus__', token)
		self.pools.unlock()

	def guess(self, msg):
		tokens = Set(self.getTokens(msg))
		return self.pools.guessProbs(tokens, self.combiner)

	def robinson(self, probs, ignore):
		""" computes the probability of a message being spam (Robinson's method)
			P = 1 - prod(1-p)^(1/n)
			Q = 1 - prod(p)^(1/n)
			S = (1 + (P-Q)/(P+Q)) / 2
			Courtesy of http://christophe.delord.free.fr/en/index.html
		"""
		nth = 1./len(probs)
		P = 1.0 - reduce(operator.mul, map(lambda p: 1.0-p[1], probs), 1.0) ** nth
		Q = 1.0 - reduce(operator.mul, map(lambda p: p[1], probs)) ** nth
		S = (P - Q) / (P + Q)
		return (1 + S) / 2


	def robinsonFisher(self, probs, ignore):
		""" computes the probability of a message being spam (Robinson-Fisher method)
			H = C-1( -2.ln(prod(p)), 2*n )
			S = C-1( -2.ln(prod(1-p)), 2*n )
			I = (1 + H - S) / 2
			Courtesy of http://christophe.delord.free.fr/en/index.html
		"""
		n = len(probs)
		try: H = chi2P(-2.0 * math.log(reduce(operator.mul, map(lambda p: p[1], probs), 1.0)), 2*n)
		except OverflowError: H = 0.0
		try: S = chi2P(-2.0 * math.log(reduce(operator.mul, map(lambda p: 1.0-p[1], probs), 1.0)), 2*n)
		except OverflowError: S = 0.0
		return (1 + H - S) / 2

class Tokenizer:
	"""A simple regex-based whitespace tokenizer.
	It expects a string and can return all tokens lower-cased
	or in their existing case.
	"""
	WORD_RE = re.compile(r'\w+', re.U)

	def __init__(self, lower=False):
		self.lower = lower

	def tokenize(self, obj):
		for match in self.WORD_RE.finditer(obj):
			if self.lower:
				yield match.group().lower()
			else:
				yield match.group()


class NGrams:
	def __init__(self, lower=False):
		self.lower = lower

	def tokenize(self, obj):
		n = 5
		for i in range(len(obj) - n + 1):
			yield obj[i:i + n]

def chi2P(chi, df):
	""" return P(chisq >= chi, with df degree of freedom)

	df must be even
	"""
	assert df & 1 == 0
	m = chi / 2.0
	sum = term = math.exp(-m)
	for i in range(1, df/2):
		term *= m/i
		sum += term
	return min(sum, 1.0)

## BAYES END

def parse_args(args=None):
	parser = argparse.ArgumentParser(description='Who wrote this shit?')
	parser.add_argument('file', nargs='+', help='Input file')
	parser.add_argument('-T', '--tag', help='Tag, e.g. \'Thaddeus T. Grugq\' or \'@thegrugq\'')
	parser.add_argument('-t', '--train', action='store_true', help='Training mode')
	parser.add_argument('-u', '--untrain', action='store_true', help='Untraining mode')
	parser.add_argument('-g', '--guess', action='store_true', help='Guessing mode')
	parser.add_argument('-C', '--convert', help='Treat input file as a DB and convert it to the given file (of another type).')
	parser.add_argument('--train-dir', help='Root directory for WWTS train files. Default is {0}.'.format(app.Config.TRAIN_ROOT_DIR))
	args = parser.parse_args(args)
	return args

def guess_db_type(filename):
	_, ext = os.path.splitext(filename)
	if ext == '.pkl':
		return Pools
	elif ext == '.sqlite':
		return SQLPools
	else:
		raise RuntimeError('Cannot guess DB type from file: {0}'.format(filename))

def stdout_traceline(line):
	sys.stdout.write(line)
	sys.stdout.write('\r')

def read_file(filename):
	with codecs.open(filename, 'r', encoding='utf-8') as content:
		text = content.read()
	_, ext = os.path.splitext(filename)
	if ext == '.html':
		# Clean HTML doc from non-user-content tags,
		# and extract only text content.
		import bs4
		soup = bs4.BeautifulSoup(text, "lxml")
		drop_tags = [
				'script', 'noscript',
				]
		for tag in drop_tags:
			for element in soup.find_all(tag):
				element.extract()
		for element in soup.find_all('div', style='display:none'):
			element.extract()
		return soup.text
	return text

def run_wwts(args):
	config = app.Config(
			TRAIN_ROOT_DIR=args.train_dir,
			)

	if args.convert:
		print('Converting {0} -> {1}'.format(args.file[0], args.convert))
		type_src = guess_db_type(args.file[0])
		type_dest = guess_db_type(args.convert)

		src_db = type_src(config.TRAIN_ROOT_DIR)
		src_db.load(args.file[0])

		dest_db = type_dest(config.TRAIN_ROOT_DIR)
		dest_db.load(args.convert)

		dest_db.import_from(src_db, tracer=stdout_traceline)
		sys.stdout.write('\n') # To preserve currently displayed traceline.
		dest_db.save(args.convert)

		print('Verifying...')
		src_pools = set(src_db.poolNames())
		dest_pools = set(dest_db.poolNames())
		if src_pools != dest_pools:
			print('List of pools differs:')
			print('  src : {0}'.format(src_pools))
			print('  dest: {0}'.format(dest_pools))
			return 1
		for name in src_db.poolNames() + ['__Corpus__']:
			src_iter = iter(src_db.iter_pool_words(name))
			dest_iter = iter(dest_db.iter_pool_words(name))
			while True:
				src_data = next(src_iter, None)
				dest_data = next(dest_iter, None)
				if src_data is None and dest_data is None:
					break
				if src_data != dest_data:
					print('Pool {0}: tokens differ:'.format(name))
					print('  src : {0}'.format(src_data))
					print('  dest: {0}'.format(dest_data))
					return 1
		print('Done.')
		return

	wwts = Bayes(config, tokenizer=Tokenizer(lower=True))
	try:
		wwts.load()
	except Exception as e:
		traceback.print_exc()
		if args.train or args.untrain:
			pass
		elif args.guess:
			parser.error('cannot load identity file')

	#print(args)
	#print(args.train, args.untrain)
	if args.train:
		if not args.tag:
			parser.error('argument -T/--tag is required.')
		for f in args.file:
			if os.path.isdir(f):
				continue
			wwts.train(args.tag, read_file(f))
			wwts.save()
			print("Trained {0}".format(f))
	elif args.untrain:
		if not args.tag:
			parser.error('argument -T/--tag is required.')
		for f in args.file:
			if os.path.isdir(f):
				continue
			wwts.untrain(args.tag, read_file(f))
			wwts.save()
			print("Untrained {0}".format(f))
	elif args.guess:
		for f in args.file:
			if os.path.isdir(f):
				continue
			print(str(wwts.guess(read_file(f))) + " " + f)
	else:
		parser.error('argument -t/--train or -g/--guess is required')

def main(args=None):
	try:
		run_wwts(args)
		return 0
	except KeyboardInterrupt as e:
		raise e
	except SystemExit as e:
		raise e
	except Exception as e:
		print(e)
		traceback.print_exc()
		return 1

# Fixing old issue with compatibility of pickle dicts.
sys.modules['__main__'].BayesData = BayesData

def wwts_guess():
	parser = argparse.ArgumentParser(description='Who wrote this shit?')
	parser.add_argument('--train-dir', help='Root directory for WWTS train files. Default is {0}.'.format(app.Config.TRAIN_ROOT_DIR))
	parser.add_argument('file', nargs='+', help='Input file')
	args = parser.parse_args()

	args.guess = True
	args.convert = False
	return main(args)

def wwts_train():
	parser = argparse.ArgumentParser(description='Who wrote this shit?')
	parser.add_argument('--train-dir', help='Root directory for WWTS train files. Default is {0}.'.format(app.Config.TRAIN_ROOT_DIR))
	parser.add_argument('--dest-dir', help='Directory to store downloaded feeds. Default is {0}.'.format(app.Config.RSS_DIR))
	parser.add_argument('tag', nargs=1, help='New tag')
	parser.add_argument('file', nargs='+', help='Input file')
	args = parser.parse_args()

	import os, shutil
	ROOT_DIR = args.dest_dir or app.Config.RSS_DIR
	tag = args.tag = args.tag[0]
	if len(args.file) < 1:
		print("No filenames provided!")
		return 1
	args.train = True
	args.convert = False
	dirname = tag.replace('good', 'other').replace('bad', 'unwanted')
	#if [ "x$TAG" == "xgood" ]; then
	if 0 == main(args):
		if not os.path.exists(os.path.join(ROOT_DIR, dirname)):
			os.makedirs(os.path.join(ROOT_DIR, dirname))
		for name in args.file:
			shutil.move(name, os.path.join(ROOT_DIR, dirname))
	#elif [ "x$TAG" == "xbad" ]; then
		#wwts -F "$@" -u -T "good" && mkdir -p "$ROOT_DIR/$TAG" && mv -- "$@" "$ROOT_DIR/$TAG"
	#else
		#echo "Unknown tag ${TAG}. Must be 'good' or 'bad'."
	#fi

if __name__ == '__main__':
	args = parse_args()
	sys.exit(main(args))
