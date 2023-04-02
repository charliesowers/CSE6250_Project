import os
import pickle
import pandas as pd
import numpy as np
import sqlite3

mincount = 100

df = pd.read_csv("./S1_File.txt", sep='\t', header=0)

countsdf = df['DX_GROUP_DESCRIPTION'].value_counts().reset_index()

mincountdf = countsdf[countsdf['DX_GROUP_DESCRIPTION'] > mincount]

# create map of words to features
codemap = {}
for i in range(len(mincountdf)):
 	codemap[mincountdf['index'][i]] = i+2
	 
inpatdf = df[df["SERVICE_LOCATION"] == 'INPATIENT HOSPITAL']
inpatdf['INPAT_ID'] = inpatdf["DAY_ID"]
inpatdf = inpatdf[['PID', 'INPAT_ID']].drop_duplicates().sort_values(by=['PID', 'INPAT_ID'])



df['feature'] = df.apply(lambda x: codemap[x['DX_GROUP_DESCRIPTION']] if x['DX_GROUP_DESCRIPTION'] in codemap else 1, axis=1)

df_grp = df[['PID', 'DAY_ID', 'feature']]
df_grp = df_grp.drop_duplicates().sort_values(by=['PID', 'DAY_ID','feature'])

grouped1 = df_grp.groupby(['PID', 'DAY_ID']).agg({'feature':lambda x: list(x)}).reset_index()
grouped2 = grouped1.groupby(['PID']).agg({'feature':lambda x: list(x)}).reset_index()
patient_ids = grouped2['PID'].tolist()
seqs = grouped2['feature'].tolist()

visitsdf = grouped1[['PID', 'DAY_ID']]
conn = sqlite3.connect(':memory:')
#write the tables
inpatdf.to_sql('inpat', conn, index=False)
visitsdf.to_sql('visits', conn, index=False)

qry = '''
    select  
        visits.PID, visits.DAY_ID,
        inpat.INPAT_ID
    from
        visits left join inpat on
		visits.DAY_ID < inpat.INPAT_ID and inpat.INPAT_ID <= visits.DAY_ID+30 and visits.PID = inpat.PID
    '''
vis_inpat = pd.read_sql_query(qry, conn)
vis_inpat['LABEL'] = vis_inpat.apply(lambda x: 0 if np.isnan(x['INPAT_ID']) else 1, axis=1)
vis_inpat = vis_inpat[['PID', 'DAY_ID', 'LABEL']].drop_duplicates().sort_values(by=['PID', 'DAY_ID'])
label_grp = vis_inpat.groupby(['PID']).agg({'LABEL':lambda x: list(x)}).reset_index()
labels = label_grp['LABEL'].tolist()

# varaible seqs contains list of lists of lists of ints corresponding to features in each visit for each patient
#variable label contains list of lists of 0/1 corresponding to if a visit required a 30-day readmission
