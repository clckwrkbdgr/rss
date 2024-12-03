#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
import traceback
import re
import codecs
from functools import reduce
import pickle
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
		self.training = []
		self.pool = None
		self.tokenCount = 0
		self.trainCount = 0

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
	def get_pool_tokenCount(self, name):
		return self.pools.get(name).tokenCount
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
	def pool_dec_token(self, pool_name, token):
		pool = self.pools[pool_name]
		count = pool.get(token, 0)
		if count:
			if count == 1:
				del(pool[token])
			else:
				pool[token] =  count - 1
			self.adjust_token_count(pool_name, -1)
	def has_pool(self, name):
		return name in self.pools
	def add_training(self, pool_name):
		self.pools[pool_name].trainCount += 1
	def adjust_token_count(self, pool_name, wc):
		self.pools[pool_name].tokenCount += wc
	def add_trained_uid(self, pool_name, uid):
		self.pools[pool_name].training.append(uid)
	def remove_trained_uid(self, pool_name, uid):
		self.pools[pool_name].training.remove(uid)

	def newPool(self, poolName):
		"""Create a new pool, without actually doing any
		training.
		"""
		self.pools.setdefault(poolName, self.dataClass(poolName))
	def removePool(self, poolName):
		del(self.pools[poolName])
	def renamePool(self, poolName, newName):
		self.pools[newName] = self.pools[poolName]
		self.pools[newName].name = newName
		self.removePool(poolName)
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
				dp.tokenCount += 1
	def save(self, fname='train.pkl'):
		fp = open(os.path.join(self.store_dir, fname), 'wb')
		pickle.dump(self.pools, fp)
		fp.close()
	def load(self, fname='train.pkl'):
		fp = open(os.path.join(self.store_dir, fname), 'rb')
		self.pools = pickle.load(fp)
		fp.close()
	def poolNames(self):
		"""Return a sorted list of Pool names.
		Does not include the system pool '__Corpus__'.
		"""
		pools = list(self.pools.keys())
		pools.remove('__Corpus__')
		pools = [pool for pool in pools]
		pools.sort()
		return pools

class Bayes(object):

	def __init__(self, config, tokenizer=None, combiner=None, dataClass=None):
		self.config = config
		if dataClass is None:
			self.dataClass = BayesData
		else:
			self.dataClass = dataClass
		self.pools = Pools(self.config.TRAIN_ROOT_DIR, dataClass=dataClass)
		self.dirty = True
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
		self.dirty = True # not always true, but it's simple
		self.pools.newPool(poolName)

	def removePool(self, poolName):
		self.pools.removePool(poolName)
		self.dirty = True

	def renamePool(self, poolName, newName):
		self.pools.renamePool( poolName, newName)
		self.dirty = True

	def mergePools(self, destPool, sourcePool):
		self.pools.mergePools(destPool, sourcePool)
		self.dirty = True

	def save(self, fname='train.pkl'):
		self.pools.save()

	def load(self, fname='train.pkl'):
		self.pools.load()
		self.dirty = True

	def poolNames(self):
		return self.pools.poolNames()

	def buildCache(self):
		""" merges corpora and computes probabilities
		"""
		self.cache = {} # FIXME also merge with Pools as a separate DB instance for Cache.
		for pname in self.pools.poolNames():
			poolCount = self.pools.get_pool_tokenCount(pname)
			themCount = max(self.pools.get_pool_tokenCount('__Corpus__') - poolCount, 1)
			cacheDict = self.cache.setdefault(pname, self.dataClass(pname))

			for word, totCount, thisCount in self.pools.iter_pool_corpus_words(pname):
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
					cacheDict[word] = max(0.0001, min(0.9999, f))

	def poolProbs(self):
		if self.dirty:
			self.buildCache()
			self.dirty = False
		return self.cache

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

	def getProbs(self, pool, words):
		""" extracts the probabilities of tokens in a message
		"""
		probs = [(word, pool[word]) for word in words if word in pool]
		probs.sort(key=lambda x: x[1], reverse=True)
		return probs[:2048]

	def train(self, pool_name, item, uid=None):
		"""Train Bayes by telling him that item belongs
		in pool. uid is optional and may be used to uniquely
		identify the item that is being trained on.
		"""
		tokens = self.getTokens(item)
		self.pools.newPool(pool_name)
		self._train(tokens, pool_name)
		self.pools.add_training('__Corpus__')
		self.pools.add_training(pool_name)
		if uid:
			self.pools.add_trained_uid(pool_name, uid)
		self.dirty = True

	def untrain(self, pool_name, item, uid=None):
		tokens = self.getTokens(item)
		if not self.pools.has_pool(pool_name):
			return
		self._untrain(tokens, pool_name)
		# I guess we want to count this as additional training?
		self.pools.add_training('__Corpus__')
		self.pools.add_training(pool_name)
		if uid:
			self.pools.remove_trained_uid(pool_name, uid)
		self.dirty = True

	def _train(self, tokens, pool_name):
		wc = 0
		for token in tokens:
			self.pools.pool_inc_token(pool_name, token)
			self.pools.pool_inc_token('__Corpus__', token)
			wc += 1
		self.pools.adjust_token_count(pool_name, wc)
		self.pools.adjust_token_count('__Corpus__', wc)

	def _untrain(self, tokens, pool_name):
		for token in tokens:
			self.pools.pool_dec_token(pool_name, token)
			self.pools.pool_dec_token('__Corpus__', token)

	def guess(self, msg):
		tokens = Set(self.getTokens(msg))
		pools = self.poolProbs()

		res = {}
		for pname, pprobs in pools.items():
			p = self.getProbs(pprobs, tokens)
			if len(p) != 0:
				res[pname]=self.combiner(p, pname)
		res = list(res.items())
		res.sort(key=lambda x: x[1], reverse=True)
		return res

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
	parser.add_argument('--train-dir', help='Root directory for WWTS train files. Default is {0}.'.format(app.Config.TRAIN_ROOT_DIR))
	args = parser.parse_args(args)
	return args

def run_wwts(args):
	config = app.Config(
			TRAIN_ROOT_DIR=args.train_dir,
			)

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
			with codecs.open(f, 'r', encoding='utf-8') as content:
				wwts.train(args.tag, content.read())
				wwts.save()
				print("Trained {0}".format(f))
	elif args.untrain:
		if not args.tag:
			parser.error('argument -T/--tag is required.')
		for f in args.file:
			if os.path.isdir(f):
				continue
			with codecs.open(f, 'r', encoding='utf-8') as content:
				wwts.untrain(args.tag, content.read())
				wwts.save()
				print("Untrained {0}".format(f))
	elif args.guess:
		for f in args.file:
			if os.path.isdir(f):
				continue
			with codecs.open(f, 'r', encoding='utf-8') as content:
				print(str(wwts.guess(content.read())) + " " + f)
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
