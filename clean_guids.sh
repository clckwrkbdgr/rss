#!/bin/bash

ENTRIES_TO_KEEP=150
SQLFILE=$(mktemp)

echo 'select feed from Guids group by feed having count(*) > 100;' | sqlite3 ~/.guids.sqlite | while read FEED; do
	echo ' delete from Guids '               >>"$SQLFILE"
	echo " where feed = \"$FEED\" "          >>"$SQLFILE"
	echo " and datetime < ( "                >>"$SQLFILE"
	echo " select datetime from Guids "      >>"$SQLFILE"
	echo " where feed = \"$FEED\" "          >>"$SQLFILE"
	echo " order by datetime desc "          >>"$SQLFILE"
	echo " limit 1 offset $ENTRIES_TO_KEEP " >>"$SQLFILE"
	echo " ); "                              >>"$SQLFILE"
done
echo " vacuum; "                         >>"$SQLFILE"

cat "$SQLFILE" | sqlite3 ~/.guids.sqlite
