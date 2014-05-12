#!/usr/bin/python3
import sqlite3
import datetime

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


