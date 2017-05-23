#!/usr/bin/env python2

import os
import logging
import urllib2
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import ast
import re
import json

# from compensation import Compensator

logging.basicConfig(level="INFO")
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data/{}/human_raw")

def to_snake_case(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub(r'[.:\/]', '_', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


class Labeler(object):
    """Assigns unique integer labels."""
    def __init__(self, init=()):
        self._labels = {}
        self._xs = []
        for x in init:
            self.label(x)

    def label(self, x):
        if x not in self._labels:
            self._labels[x] = len(self._labels)
            self._xs.append(x)
        return self._labels[x]

    def unlabel(self, label):
        return self._xs[label]

    __call__ = label

def add_auth(url, username, password):
    """Add HTTP authencation for opening urls with urllib2.

    Based on http://www.voidspace.org.uk/python/articles/authentication.shtml

    """

    # this creates a password manager
    passman = urllib2.HTTPPasswordMgrWithDefaultRealm()

    # because we have put None at the start it will always use this
    # username/password combination for urls for which `theurl` is a
    # super-url
    passman.add_password(None, url, username, password)

    # create the AuthHandler
    authhandler = urllib2.HTTPBasicAuthHandler(passman)

    # All calls to urllib2.urlopen will now use our handler Make sure
    # not to include the protocol in with the URL, or
    # HTTPPasswordMgrWithDefaultRealm will be very confused.  You must
    # (of course) use it when fetching the page though.
    opener = urllib2.build_opener(authhandler)
    urllib2.install_opener(opener)


def fetch(site_root, filename, version, experiment=1, force=True):
    """Download `filename` from `site_root` and save it in the
    human-raw/`experiment` data folder.

    """

    # get the url
    url = os.path.join(site_root, version, filename)

    # get the destination to save the data, and don't do anything if
    # it exists already
    dest = os.path.join(DATA_PATH.format(experiment), version, "%s.csv" % os.path.splitext(filename)[0])
    if os.path.exists(dest) and not force:
        print('{} already exists. Use --force to overwrite.'.format(dest))
        return

    # try to open it
    try:
        handler = urllib2.urlopen(url)
    except IOError as err:
        if getattr(err, 'code', None) == 401:
            logging.error("Server authentication failed.")
            raise err
        else:
            raise

    # download the data
    data = handler.read()
    logging.info("Fetched succesfully: %s", url)

    # make the destination folder if it doesn't exist
    if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))

    # write out the data file
    with open(dest, "w") as fh:
        fh.write(data)
    logging.info("Saved to '%s'", os.path.relpath(dest))
    if filename == 'questiondata':
        df = pd.read_csv(dest, header=None)
        n_pid = df[0].unique().shape[0]
        logging.info('Number of participants: %s', n_pid)


def reformat_data(experiment, version):
    data_path = 'data/{experiment}/human_raw/{version}/'.format(
        experiment=experiment,
        version=version,
    )

    # Create participants dataframe (pdf)
    def parse_questiondata():
        qdf = pd.read_csv(data_path + 'questiondata.csv', header=None)
        for uid, df in qdf.groupby(0):
            row = ast.literal_eval(list(df[df[1] == 'params'][2])[0])
            row['worker_id'], row['assignment_id'] = uid.split(':')

            bonus_row = df[df[1] == 'final_bonus']
            if len(bonus_row):
                bonus = float(list(bonus_row[2])[0])
                row['bonus'] = bonus
                row['completed'] = True
            else:
                row['bonus'] = 0
                row['completed'] = False
            yield row

    pdf = pd.DataFrame(parse_questiondata())
    pdf['version'] = version

    # Create trials dataframe (tdf)
    def parse_trialdata():
        tdf = pd.read_csv(data_path + 'trialdata.csv', header=None)
        tdf = pd.DataFrame.from_records(tdf[3].apply(json.loads)).join(tdf[0])
        tdf['worker_id'] = tdf[0].apply(lambda x: x.split(':')[0])
        tdf['assignment_id'] = tdf[0].apply(lambda x: x.split(':')[1])
        # df = df[df.worker_id.map(lambda x: not x.startswith('debug'))].copy()
        return tdf.drop(0, axis=1)

    tdf = parse_trialdata()


    pid_labeler = Labeler()
    pdf['pid'] = pdf['worker_id'].apply(pid_labeler)
    tdf['pid'] = tdf['worker_id'].apply(pid_labeler)


    data = {'participants': pdf}
    for trial_type, df in tdf.groupby('trial_type'):
        # df = df.dropna(axis=1)
        df = df.drop('internal_node_id', axis=1)
        df = df.drop('trial_index', axis=1)
        df.columns = [to_snake_case(c) for c in df.columns]
        data[trial_type] = df
    return data


def write_data(experiment, code_version, data):
    path = 'data/{experiment}/human/{code_version}/'.format(
        experiment=experiment,
        code_version=code_version,
    )
    if not os.path.isdir(path):
        os.makedirs(path)
    for name, df in data.items():
        dest = path + name + '.csv'
        df.to_csv(dest, index=False)
        print('wrote {} with {} rows.'.format(dest, len(df)))

# def compensate(pdf):
#     print('BONUS   mean={}, max={}, total={}'
#           .format(pdf.bonus.mean(), pdf.bonus.max(), pdf.bonus.sum()))
#     if input('Approve? y/[n]') != 'y':
#         print('Not assigning bonuses')
#     else:
#         comp = Compensator(verbose=False)
#         for i, row in pdf.iterrows():
#             comp.approve(row.assignment_id)
#             comp.grant_bonus(row.worker_id, row.assignment_id, round(row.bonus, 2))

def main(version, 
         address='http://cocosci-fred.dreamhosters.com/data',
         username='fredcallaway',
         password='cocotastic90'):

    add_auth(address, username, password)
    files = ["trialdata", "eventdata", "questiondata"]
    for filename in files:
        fetch(address, filename, version)
    data = reformat_data(1, version)
    write_data(1, version, data)


if __name__ == "__main__":
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-v", "--version",
        required=True,
        help=("Experiment version. This corresponds to the experiment_code_version "
              "parameter in the psiTurk config.txt file that was used when the "
              "data was collected."))
    parser.add_argument(
        "-a", "--address",
        default="http://cocosci-fred.dreamhosters.com/data",
        help="Address from which to fetch data files.")
    parser.add_argument(
        "-u", "--user",
        default='fredcallaway',
        help="Username to authenticate to the server.")
    parser.add_argument(
        "-p", "--password",
        default='cocotastic90',
        help="Password to authenticate to the server.")

    args = parser.parse_args()
    main(args.version, args.address, args.user, args.password)



# file = 'data/1/human_raw/I/eventdata.csv'
# edf = pd.read_csv(file, header=None)

# wid_to_pid = dict(pdf.set_index('worker_id').pid)
# def get_pid(x):
#     wid = x.split(':')[0]
#     return wid_to_pid.get(wid, -1)

# edf = pd.DataFrame({
#     'pid': edf[0].apply(get_pid),
#     'kind': edf[1],
#     'time': edf[4],
#     'time_diff': edf[2],
#     'arg': edf[3],
# }).set_index('pid', drop=False)