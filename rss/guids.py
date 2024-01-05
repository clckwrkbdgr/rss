#!/usr/bin/python3
import os
import sqlite3
import datetime
from . import app

def _now():
	return datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%dT%H%M%S%f")

class GuidDatabase:
	def __init__(self, filename):
		self.conn = sqlite3.connect(filename)
		self.conn.text_factory = str # To prevent some dummy encoding bug.
		self.c = self.conn.cursor()
		self.c.execute("""create table if not exists Guids (feed text, guid text, datetime text);""")
		self.conn.commit()
	
	def close(self):
		self.c.close()
		self.conn.close()
	
	def add_guid(self, feed, guid):
		self.c.execute("""insert into Guids values (?, ?, ?);""", (feed, guid, _now()))
		self.conn.commit()
	
	def guid_exists(self, feed, guid):
		self.c.execute("""select count(*) from Guids where feed=? and guid=?;""", (feed, guid))
		self.conn.commit()
		count = [int(f) for f, in self.c]
		return count[0] if count else 0

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
