import pandas as pd
from utils import *
from toolz.curried import *
from ast import literal_eval

def load(v, base='../experiments/data/human'):
    """Loads data from experiment with codeversion `v`."""
    df = pd.read_csv(f'{base}/{v}/mouselab-mdp.csv')
    pdf = pd.read_csv(f'{base}/{v}/participants.csv')

    pr_types = 'none objectLevel featureBased'.split()
    pdf['pr_type'] = pdf.PR_type.astype('category', categories=pr_types)
    pdf.drop('PR_type', axis=1, inplace=True)

    completed = list(pdf.query('completed').pid)
    df = df.query('pid == @completed').copy()
    pdf = pdf.query('pid == @completed').copy()

    df.trial_index = df.trial_index.astype(int) - 1
    df.trial_id = df.trial_id.astype(int)
    df['version'] = list(pdf.version.loc[df.pid])
    df['info_cost'] = list(pdf.info_cost.loc[df.pid])
    df['pr_type'] = pd.Categorical(pdf.pr_type.loc[df.pid], categories=pr_types)
    df['message'] = list(pdf.message.loc[df.pid])


    df.queries = df.queries.apply(literal_eval)
    df['clicks'] = df.queries.apply(
        lambda x: lmap(int, x['click']['state']['target'])
    )
    df['click_times'] = df.queries.apply(
        lambda x: x['click']['state']['time']
    )
    df.action_times = df.action_times.apply(literal_eval)
    df['n_click'] = df.clicks.apply(len)
    df.delays = df.delays.apply(literal_eval)
    df.path = df.path.apply(
        lambda x: [1] + literal_eval(x)[1:])
    df['feedback'] = df.planned_too_much.apply(literal_eval).apply(len)

    pdf['time_elapsed'] = df.groupby('pid').time_elapsed.max() / 60000
    pdf['score'] = df.groupby('pid').score.sum()
    df.actions = df.actions.apply(literal_eval)
    df['states'] = df.pop('beliefs').apply(literal_eval)

    # df = df.query('info_cost == 1.25').copy()
    # pdf = pdf.query('info_cost == 1.25').copy()
    labeler = Labeler()
    df['pidx'] = df.pid.apply(labeler)
    pdf['pidx'] = pdf.pid.apply(labeler)
    df = df.rename(columns={'p_rdata': 'pr_data'})
    df.pr_data = df.pr_data.fillna('None').apply(literal_eval)

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

