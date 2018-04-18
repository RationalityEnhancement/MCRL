#!/usr/bin/env python3
import sys
import pandas as pd
from analysis_utils import get_data

data = get_data(sys.argv[1])
pdf = data['participants']
ids = pd.read_csv('data/human_raw/I0.9/identifiers.csv')
pdf['worker_id'] = ids.worker_id
pdf.set_index('worker_id').bonus.to_json('stage1.json')