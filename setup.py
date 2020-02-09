import setuptools
import re, subprocess
version = '.'.join(sorted([
		m.group(1).split('.')
		for m in
		(re.match(r'^.*(\d+[.]\d+[.]\d+)$', line) for line in subprocess.check_output(['git', 'tag']).decode().splitlines())
		if m
		])[-1])
setuptools.setup(
		name='rss',
		version=version,
		packages=['rss'],
		entry_points={
			"console_scripts" : [
				'urss = rss.rss:main',
				'wwts = rss.wwts:main',
				'wwts_guess = rss.wwts:wwts_guess',
				'wwts_train = rss.wwts:wwts_train',
				'clean_urss_guids = rss.guids:clean_guids',
				]
			},
		)

