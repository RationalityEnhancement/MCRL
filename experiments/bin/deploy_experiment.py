#!/usr/bin/env python3

import os
import subprocess

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from termcolor import colored



def upload(args):
    address = "%s@%s" % (args.user, args.host)
    root_path = "%s:/home/cocosci/cocosci-mcrl.dreamhosters.com/" % (address)
    deploy_path = "%s:%s" % (address, args.dest)

    src_paths = [
        os.path.relpath(os.path.join(args.path, "passenger_wsgi.py")),
        os.path.relpath(os.path.join(args.path, "static")),
        os.path.relpath(os.path.join(args.path, "templates")),
        os.path.relpath(os.path.join(args.path, "remote-config.txt")),
        os.path.relpath(os.path.join(args.path, "custom.py"))
    ]
    dst_paths = [
        root_path,
        deploy_path,
        deploy_path,
        os.path.join(deploy_path, "config.txt"),
        deploy_path
    ]

    cmd_template = ["rsync", "-av", "--delete-after", "--copy-links"]
    if args.dry_run:
        cmd_template.append("-n")
    if args.bwlimit:
        cmd_template.append("--bwlimit=%d" % args.bwlimit)
    cmd_template.append("%s")
    cmd_template.append("%s")

    for source, dest in zip(src_paths, dst_paths):
        cmd = " ".join(cmd_template) % (source, dest)
        print(colored(cmd, 'blue'))
        code = subprocess.call(cmd, shell=True)
        if code != 0:
            raise RuntimeError("rsync exited abnormally: %d" % code)

    cmd = ("ssh %s "
           "'rm -f /home/cocosci/cocosci-mcrl.dreamhosters.com/tmp/restart.txt && "
           "touch /home/cocosci/cocosci-mcrl.dreamhosters.com/tmp/restart.txt'" % (
               address))

    print(colored(cmd, 'blue'))
    code = subprocess.call(cmd, shell=True)
    if code != 0:
        raise RuntimeError("command exited abnormally: %s" % code)


def archive(args):
    if not args.archive:
        return
    import re
    with open('{}/remote-config.txt'.format(args.path)) as f:
        version = re.search('experiment_code_version = (\w+)', f.read()).group(1)

        dest = '.archive/%s' % version
        if os.path.isdir(dest):
            ok = input('Already deployed version %s. Continue? y/[n]  '
                       % version).startswith('y')
            if not ok:
                print('Exiting.')
                exit(1)
        
        os.makedirs(dest, exist_ok=True)
        cmd = 'rsync -av --delete-after --copy-links %s %s' % (args.path, dest)
        print(colored(cmd, 'blue'))
        subprocess.call(cmd, shell=True)


def check_debug(args):
    with open('{}/static/js/experiment.js'.format(args.path)) as f:
        if 'DEBUG = true' in f.read():
            ok = input('DEBUG flag is set in experiment.js. Continue? y/[n]  ').startswith('y')
            if not ok:
                print('Exiting.')
                exit(1)


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'path',
        help='local path of exeriment')
    parser.add_argument(
        "-u", "--user",
        default="cocosci",
        help="Username to login to the server.")
    parser.add_argument(
        "--host",
        default="cocosci-mcrl.dreamhosters.com",
        help="Hostname of the experiment server.")
    parser.add_argument(
        "-n", "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="Show what would have been transferred.")
    parser.add_argument(
        "--bwlimit",
        type=int,
        default=2000,
        help="Bandwidth limit for transfer")
    parser.add_argument(
        "-dest",
        default="/home/cocosci/cocosci-mcrl.dreamhosters.com/experiment",
        help="Destination path on the experiment server.")
    parser.add_argument(
        "-a", '--archive',
        default=False,
        type=bool,
        help="Do not archive experiment/.")

    args = parser.parse_args()
    check_debug(args)
    archive(args)
    upload(args)
