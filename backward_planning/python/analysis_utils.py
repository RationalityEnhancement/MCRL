import numpy as np
import pandas as pd
import re
import os

from utils import *

# ---------- Data wrangling ---------- #

def mostly_nan(col):
    try:
        return col.apply(np.isnan).mean() > 0.5
    except:
        return False

def drop_nan_cols(df):
    return df[[name for name, col in df.iteritems()
               if not mostly_nan(col)]]

def query_subset(df, col, subset):
    idx = df[col].apply(lambda x: x in subset)
    return df[idx].copy()

def rowapply(df, f):
    return [f(row) for i, row in df.iterrows()]

def to_snake_case(name):
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub(r'[.:\/]', '_', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def to_camel_case(snake_str):
    return ''.join(x.title() for x in snake_str.split('_'))

def reformat_name(name):
    return re.sub('\W', '', to_snake_case(name))


# ---------- Loading data ---------- #

from glob import glob
import json
import ast
def parse_json(df):
    def can_eval(x):
        try:
            ast.literal_eval(x)
            return True
        except:
            return False


    to_eval = df.columns[df.iloc[0].apply(can_eval)]
    for col in to_eval:
        try:
            df[col] = df[col].apply(ast.literal_eval)
        except:
            pass

def get_data(version, data_path='data'): 
    data = {}
    for file in glob('{}/human/{}/*.csv'
                     .format(data_path, version)):
        name = os.path.basename(file)[:-4]
        df = pd.read_csv(file)
        parse_json(df)
        data[name] = drop_nan_cols(df)




    # n_trials = df.pid.value_counts().max()
    # complete = df.pid.value_counts(sort=False).where(lambda x: x==n_trials).dropna().index
    # df = df.set_index('pid').ix[complete].reset_index()
    # pdf = pdf.ix[complete]
    
    return data

@curry
def load(file, version=None, func=lambda x: x):
    if not file or type(file) == float:
        return None
    else:
        base = '.archive/{}/'.format(version) if version else ''
        with open('{}experiment/{}'.format(base, file)) as f: 
            return func(json.load(f))


# ---------- Statistics ---------- #

try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()
    from rpy2.robjects.conversion import ri2py
except:
    print("Error importing rpy2")
else:
    def r2py(results, p_col=None):
        tbl = ri2py(results)
        tbl = tbl.rename(columns=reformat_name)
        if p_col:
            tbl['signif'] = tbl[reformat_name(p_col)].apply(pval)
        return tbl

def df2r(df, cols):
    df = df[cols].copy()
    for name, col in df.iteritems():
        if col.dtype == bool:
            df[name] = col.astype(int)
    return df

def pval(x):
    if x < 0.001:
        return "p\,<\,0.001"
    elif x < 0.01:
        return "p\,<\,0.01"
    elif x < 0.05:
        return "p\,<\,0.05"
    elif x >= 0.05:
        return "p\,=\,{:.2f}".format(x)
    else:
        return float('nan')

# ---------- Saving results ---------- #



class Tex:
    chi2 = r"$\chi^2({df:.0f})={chisq:.2f},\ {signif}$"

class Variables():
    """Saves variables for use in external documents."""
    def __init__(self, path='.'):
        # os.makedirs(path, exist_ok=True)
        self.path = path
        self.csv_file = os.path.join(path, 'variables.csv')
        self.sed_file = os.path.join(path, 'variables.sed')
        self.tex_file = os.path.join(path, 'variables.tex')
        self.read()

    def read(self):
        try:
            self.series = pd.Series.from_csv(self.csv_file)
        except (OSError, pd.io.common.EmptyDataError):
            self.reset()

    def reset(self):
        self.series = pd.Series()
        self.save()

    def write(self, key, val):
        self.read()
        self.series[key] = val
        self.series.to_csv(self.csv_file)
        print('{} = {}'.format(key, val))

    def save(self):
        self.series.to_csv(self.csv_file)
        with open(self.sed_file, 'w+') as f:
            for key, val in self.series.items():
                val = str(val).replace('\\', '\\\\').replace('&', '\&')
                f.write('s/`{}`/{}/g'.format(key, val) + '\n')

        with open(self.tex_file, 'w+') as f:
            for key, val in self.series.items():
                key = to_camel_case(key)
                f.write(r'\newcommand{\%s}{%s}' % (key, val) + '\n')


    def save_analysis(self, table, tex, name='', idx='{index}', display_tex=True):
        if display_tex:
            from IPython.display import Latex, display

        for i, row in table.iterrows():
            row['index'] = i
            n = name
            if idx is not None:
                n += '_' + (idx(row) if callable(idx) else idx)
            n = reformat_name(n.format_map(row)).upper()
            
            t = tex(row) if callable(tex) else tex
            t = t.format_map(row)
            
            self.write(n, t)
            if display_tex:
                display(Latex(t))
        
        self.save()


    def write_lm(self, model, var, name):
        beta = np.round(model.params[var], 2)
        se = np.round(model.bse[var], 2)
        p = model.pvalues[var]
        p_desc = pval(p)
        
        self.write_var(
            '{}_RESULT'.format(name),
            r'$\\beta = %s,\\ \\text{SE} = %s,\\ %s$' % (beta, se, p_desc)
        )

def get_rtable(results, p_col=None):
    tbl = ri2py(results)
    tbl = tbl.rename(columns=reformat_name)
    if p_col:
        tbl['signif'] = tbl[reformat_name(p_col)].apply(pval)
    return tbl

# ---------- Plotting ---------- #

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('notebook', font_scale=1.4)
sns.set_palette('deep', color_codes=True)

class Figures(object):
    """Plots and saves figures."""
    def __init__(self, path='figs/', formats=['eps']):
        self.path = path
        self.formats = formats
        os.makedirs(path, exist_ok=True)


    def savefig(self, name):
        name = name.lower()
        for fmt in self.formats:
            path = os.path.join(self.path, name + '.' + fmt)
            print(path)
            plt.savefig(path, bbox_inches='tight')
    
    def plot(self, **kwargs1):
        """Decorator that calls a plotting function and saves the result."""
        def decorator(func):        
            def wrapped(*args, **kwargs):
                kwargs.update(kwargs1)
                params = [v for v in kwargs1.values() if v is not None]
                param_str = '_' + str_join(params).rstrip('_') if params else ''
                name = func.__name__ + param_str
                if name.startswith('plot_'):
                    name = name[len('plot_'):]
                func(*args, **kwargs)
                self.savefig(name)
            wrapped()
            return wrapped
        
        return decorator
