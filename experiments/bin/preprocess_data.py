#!/usr/bin/env python3
import pandas as pd
from ast import literal_eval
from toolz.curried import *

# import sys
# sys.path.append('lib')
# from analysis_utils import *

def load(version):
    """Loads data from experiment with codeversion `version`."""
    df = pd.read_csv('data/human/{}/mouselab-mdp.csv'.format(version))
    pdf = pd.read_csv('data/human/{}/participants.csv'.format(version))
    return df, pdf

def combine(*vs):
    """Load multiple versions at once."""
    def loop():
        first_pid = 0
        for v in vs:
            df = pd.read_csv('data/human/{}/mouselab-mdp.csv'.format(v))
            pdf = pd.read_csv('data/human/{}/participants.csv'.format(v))
            df.pid += first_pid
            pdf.pid += first_pid
            first_pid += max(pdf.pid.max(), df.pid.max()) + 1
            yield df, pdf
    dfs, pdfs = zip(*loop())
    df = pd.concat(dfs).reset_index()
    pdf = pd.concat(pdfs).reset_index().set_index('pid', drop=False)
    return df, pdf


def write_trials(version):

    # Get data for participants that completed the experiment.
    df, pdf = load(version)
    completed = list(pdf.query('completed').pid)
    df = df.query('pid == @completed')
    pdf = pdf.query('pid == @completed')

    # Add participant data to the trials dataframe
    df.trial_index = df.trial_index.astype(int)
    # df.trial_i = df.trial_i.astype(int)
    df['version'] = list(pdf.version.loc[df.pid])
    df['info_cost'] = list(pdf.info_cost.loc[df.pid])
    df['PR_type'] = list(pdf.PR_type.loc[df.pid])
    df['message'] = list(pdf.message.loc[df.pid])
    df.queries = df.queries.apply(literal_eval)
    
    # Parse fields and compute summary features.
    df['clicks'] = df.queries.apply(
        lambda x: list(map(int, x['click']['state']['target']))
    )
    df['click_times'] = df.queries.apply(
        lambda x: x['click']['state']['time']
    )
    df.action_times = df.action_times.apply(literal_eval)
    df['n_click'] = df.clicks.apply(len)
    df.delays = df.delays.apply(literal_eval)
    df.path = df.path.apply(
        lambda x: [1] + literal_eval(x)[1:])

    cols = ('pid info_cost PR_type message trial_index trial_i delays '
            'score n_click clicks click_times path action_times'
            .split())
    file = 'data/human/{}/trials.csv'.format(version)
    df[cols].to_csv(file)
    print('Wrote {}'.format(file))


if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-v", "--version",
        required=True,
        help=("Experiment version. This corresponds to the experiment_code_version "
              "parameter in the psiTurk config.txt file that was used when the "
              "data was collected."))
    write_trials(parser.parse_args().version)

# from toolz import concat
# df['info_cost'] = list(pdf.info_cost[df.pid])
# def write_state_actions():
#     data = {}
#     for cost, dd in df.groupby('info_cost'):
#         data[cost] = {}
#         data[cost]['states'] = list(concat(dd.beliefs))
#         data[cost]['actions'] = list(concat(dd.meta_actions))
#     with open('../python/data/state_actions.json', 'w+') as f:
#         json.dump(data, f)
# write_state_actions()
