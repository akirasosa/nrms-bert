from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
from tqdm import tqdm

df_nrms = pd.read_parquet('../tmp/sub_nrms.pqt')
df_pop = pd.read_parquet('../tmp/sub_popularity.pqt')

df = pd.concat((df_nrms, df_pop))
df = df.sort_index()

out_path = Path('../tmp/prediction.txt')

with out_path.open(mode='w') as fp:
    for row in tqdm(df['preds'].values):
        fp.writelines(row + '\n')

with ZipFile('../tmp/prediction.txt.zip', 'w', ZIP_DEFLATED) as z:
    z.write(out_path)

print('Done')
