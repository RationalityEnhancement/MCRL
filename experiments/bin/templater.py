#!/usr/bin/env python
"""
This script replaces variables in FILES with the values
found in variables.yml. This could be used to refer to
the HIT payment in multiple files without needing to update
every file when the payment changes.

The experiment directory is copied to an output directory, deploy/
by default, where the variable substitutions are made.

By default variables are marked <# variable_name #>.
"""

import sys
import re
import os
import subprocess

# Default arguments.
OUT_DIR = 'deploy'
FILES = [
    'templates/ad.html',
    'templates/consent.html',
    'remote-config.txt'
]
VAR_RE = r'<# (\w*) #>'  # e.g. <# payment_info #>


def main(args):

    def load_var_file():
        if args.var_file.endswith('ml'):
            import yaml
            with open(args.var_file) as f:
                return yaml.load(f)
        else:
            import json
            with open(args.var_file) as f:
                return json.load(f)

    variables = load_var_file()

    def replace(match):
        key = match.group(1)
        try:
            return variables[key]
        except KeyError:
            print('ERROR: No value found for {} in {}'.format(key, args.var_file))
            exit(1)

    def render(in_file, out_file):    
        with open(in_file) as f:
            in_txt = f.read()
        out_txt = re.sub(VAR_RE, replace, in_txt)
        with open(out_file, 'w+') as f:
            f.write(out_txt)

    cmd = 'rsync -a --delete-after --copy-links {}/ {}/'.format(args.in_dir, args.out_dir)
    print(cmd)
    subprocess.call(cmd, shell=True)
    
    for file in args.files:
        render(os.path.join(args.out_dir, file), os.path.join(args.out_dir, file))

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        description='A ~very~ simple template engine.'
    )
    parser.add_argument(
        'in_dir',
        help='Directory containing experiment.'
    )
    parser.add_argument(
        '-o', '--out_dir',
        default=OUT_DIR,
        help='Directory to write results to.'
    )
    parser.add_argument(
        '-f', '--files',
        default=FILES,
        nargs='+'
    )
    parser.add_argument(
        '-v', '--var_file',
        default='variables.yml',
        help='File containing variable definitions, in yaml or json format.'
    )
    args = parser.parse_args()
    main(args)
