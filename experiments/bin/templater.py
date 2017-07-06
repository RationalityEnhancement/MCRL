#!/usr/bin/env python3
import re
import os
import yaml
import subprocess

with open('variables.yml') as f:
    lookup = yaml.load(f)

def replace(match):
    key = match.group(1)
    try:
        return lookup[key]
    except KeyError:
        print('ERROR: No value found for <# {} #>'.format(key))
        exit(1)

def render(in_file, out_file):    
    with open(in_file) as f:
        in_txt = f.read()
    out_txt = re.sub(r'<# (\w*) #>', replace, in_txt)
    with open(out_file, 'w+') as f:
        f.write(out_txt)


def main():
    subprocess.call('rsync -a --delete-after --copy-links exp1/ deploy', shell=True)
    
    for file in ['templates/ad.html', 'templates/consent.html', 'remote-config.txt']:
        render(os.path.join('deploy', file), os.path.join('deploy', file))

if __name__ == '__main__':
    main()
