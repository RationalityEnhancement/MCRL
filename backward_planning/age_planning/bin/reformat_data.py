#!/usr/bin/env python3
import pandas as pd
import json
import ast
import re
import os

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



def fetch_data(experiment, code_version):
    data_path = 'data/{experiment}/human_raw/{code_version}/'.format(
        experiment=experiment,
        code_version=code_version,
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
    pdf['code_version'] = code_version

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
        df = df.drop('Unnamed: 0', axis=1, errors='ignore')
        df.columns = [to_snake_case(c) for c in df.columns]
        data[trial_type] = df

    return data


def write_data(experiment, code_version, data):
    path = 'data/{experiment}/human/{code_version}/'.format(
        experiment=experiment,
        code_version=code_version,
    )
    os.makedirs(path, exist_ok=True)
    for name, df in data.items():
        dest = path + name + '.csv'
        df.to_csv(dest, index=False)
        print('wrote {} with {} rows.'.format(dest, len(df)))


def main(version, experiment=1):
    data = fetch_data(experiment, version)
    write_data(experiment, version, data)
    



if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '-e', '--experiment',
        default='1',
        help='Directory in data/ to put the results.')
    parser.add_argument(
        '-v', '--version',
        required=True,
        help=('Experiment version. This corresponds to the '
              'experiment_code_version parameter in the psiTurk config.txt file '
              'that was used when the data was collected.'))

    args = parser.parse_args()
    main(args.version, args.experiment)
    



# def misformatted(df):
#     def check(c):
#         return len(c.dropna()) !=138
    
#     idx =  df.groupby('pid').correct.apply(check).as_matrix()
#     return set(df.pid.unique()[idx])

# def failed_catch(df, max_fails=1):
#     fails = (df.query("kind == 'control'")
#              .groupby('pid')
#              .correct.agg(lambda c: len(c[c== False]))
#     )
#     failed_pids = fails[fails > max_fails].index
#     return set(failed_pids)

# misformat_pids = misformatted(data['ball'])
# caught_pids = failed_catch(data['ball'])
# bad_pids = misformat_pids | caught_pids

# writevar('N_CAUGHT', len(caught_pids))
# writevar('N_FORMAT', len(misformat_pids))



# caught = data['participants'].ix[caught_pids].turk_id
# with open('caught_wids.txt', 'w+') as f:
#     f.write('\n'.join(caught))



