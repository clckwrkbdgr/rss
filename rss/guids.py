#!/usr/bin/python3
import os
import sqlite3
import datetime
from . import app

def _now():
	return datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%dT%H%M%S%f")

def parse_datetime(value):
	return datetime.datetime.strptime(value, "%Y%m%dT%H%M%S%f")

class GuidDatabase:
	def __init__(self, filename):
		self.conn = sqlite3.connect(filename, check_same_thread=False) # To allow multithreading access.
		self.conn.text_factory = str # To prevent some dummy encoding bug.
		self.c = self.conn.cursor()
		self.c.execute("""create table if not exists Feeds (feed text primary key, last_fetch text);""")
		self.c.execute("""create table if not exists Guids (feed text, guid text, datetime text);""") # TODO set foreign key for Feeds<-Guids on "feed"
		self.conn.commit()
	
	def close(self):
		self.c.close()
		self.conn.close()
	
	def add_guid(self, feed, guid):
		self.c.execute("""insert into Guids values (?, ?, ?);""", (feed, guid, _now()))
		self.conn.commit()

	def mark_fetched(self, feed):
		self.c.execute("""\
				 insert into Feeds (feed, last_fetch)
				 values (?, ?)
				 on conflict(feed)
				 do update set last_fetch = excluded.last_fetch;
				 """, (feed, _now()))
		self.conn.commit()
	
	def get_last_fetch(self, feed):
		self.c.execute("""select last_fetch from Feeds where feed=?;""", (feed,))
		self.conn.commit()
		result = [parse_datetime(f) for f, in self.c]
		return result[0] if result else datetime.datetime.min
	
	def get_total_guids(self, feed):
		self.c.execute("""select count(guid) from Guids where feed=?;""", (feed,))
		self.conn.commit()
		result = [(int(f) if f else None) for f, in self.c]
		return result[0] if result else 0

	def get_all_guids(self, feed, except_guids=None):
		if except_guids:
			self.c.execute("""\
					select guid from Guids
					where feed=?
					and guid not in ({0})
					order by datetime desc
					;""".format(','.join(['?'] * len(except_guids))), (feed,) + tuple(except_guids))
		else:
			self.c.execute("""\
					select guid from Guids where feed=?
					order by datetime desc
					;""", (feed,))
		self.conn.commit()
		result = [(str(f) if f else None) for f, in self.c]
		return result or []

	def get_all_feeds(self):
		self.c.execute("""\
				select distinct feed from Guids
				;""")
		self.conn.commit()
		result = [(str(f) if f else None) for f, in self.c]
		return result or []

	def delete_feed(self, feed):
		self.c.execute("""\
				delete from Guids
				where feed=?
				;""", (feed,))
		self.conn.commit()
		total_deleted = self.c.rowcount
		return total_deleted

	def delete_items(self, feed, guids):
		total_deleted = 0
		while guids:
			batch, guids = guids[:100], guids[100:]
			self.c.execute("""\
					delete from Guids
					where feed=?
					and guid in ({0})
					;""".format(','.join(['?'] * len(batch))), (feed,) + tuple(batch))
			self.conn.commit()
			total_deleted += self.c.rowcount
		return total_deleted

	def get_last_guid(self, feed):
		self.c.execute("""select max(datetime) from Guids where feed=? and datetime is not null;""", (feed,))
		self.conn.commit()
		result = [(parse_datetime(f) if f else None) for f, in self.c]
		return result[0] if result else datetime.datetime.min
	
	def guid_exists(self, feed, guid):
		self.c.execute("""select count(*) from Guids where feed=? and guid=?;""", (feed, guid))
		self.conn.commit()
		count = [int(f) for f, in self.c]
		return count[0] if count else 0

	def get_stats(self, feed):
		""" Returns tuple (min_interval, avg_interval, max_interval, total_interval).
		"""
		self.c.execute("""\
				select datetime from Guids
				where feed=? and datetime is not null
				order by 1
				;""", (feed,))
		self.conn.commit()
		result = [(parse_datetime(f) if f else None) for f, in self.c]

		intervals = [(end - begin) for (begin, end) in zip(result[:-1], result[1:])]
		too_close = datetime.timedelta(seconds=60) # Should be enough to skip intervals from the same run that are too close together to make significant difference.
		intervals = [_ for _ in intervals if _ > too_close]
		if not intervals:
			return (datetime.timedelta(), datetime.timedelta(), datetime.timedelta(), datetime.timedelta())
		total_interval = (result[-1] - result[0])
		avg_interval = total_interval / len(intervals)
		min_interval = min(intervals)
		max_interval = max(intervals)
		return (min_interval, avg_interval, max_interval, total_interval)

def clean_guids(config, exclude=None, just_print=False):
	ENTRIES_TO_KEEP = 150
	conn = sqlite3.connect(config.GUID_FILE)
	conn.text_factory = str # To prevent some dummy encoding bug.
	c = conn.cursor()
	c.execute("""select feed from Guids group by feed having count(*) > 100;""")
	conn.commit()
	feeds = [str(f) for f, in c]
	for feed in feeds:
		if feed in exclude:
			continue
		if just_print:
			print(feed)
			continue
		query = [
				' delete from Guids ',
				" where feed = ? ",
				" and datetime < ( ",
				" select datetime from Guids ",
				" where feed = ? ",
				" order by datetime desc ",
				" limit 1 offset ? ",
				" ); ",
				]
		c.execute('\n'.join(query), (feed, feed, ENTRIES_TO_KEEP))
		conn.commit()
	if not just_print:
		c.execute('vacuum;')
	conn.commit()

import click

@click.group()
def cli():
	pass

@cli.command('clean-guids')
@click.option('--guid-file', help='GUID file. Default is {0}.'.format(app.Config.GUID_FILE))
@click.option('--dry', is_flag=True, help='Dry run. Just print feeds that will be trimmed, one for line.')
@click.option('-e', '--exclude', multiple=True, help='Exclude these feeds from trimming.')
def command_clean_guids(guid_file=None, dry=False, exclude=None):
	""" Trims GUID queue for each feed to reduce size of gUID DB. """
	config = app.Config(
		GUID_FILE=guid_file,
		)
	clean_guids(config, exclude=exclude, just_print=dry)

if __name__ == '__main__':
	cli()
