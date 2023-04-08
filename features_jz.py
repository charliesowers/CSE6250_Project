import pandas as pd
import numpy as np
import pickle


def build_code_map(df: pd.DataFrame, feature_name: str = "index") -> dict:
    """
    map a unique feature name to an unique feature id, reserve the feature id 1 for 'unknown' (diagnosis group description that didn't meet the minium threshod)

    Parameters:
    df: input data frame
    feature_name: feature name to build the feature id

    Returns:
    codemap: dictionary of {feature_name: feature_id}
    """

    codemap = {val: index + 2 for index, val in enumerate(df[feature_name].unique())}
    codemap['Unknown'] = 1
    return codemap


def transform(df: pd.DataFrame, diagnosis_threshold: int = 100) -> (list, list):
    """
    split the data to seqs, labels

    Parameters:
    df: input data frame
    diagnosis_threshold: keep diagnosis group description that occurs more than the threshold

    Returns:
    seqs: list of patients, which is a list of diagnosis descriptions at each visit date
    labels: list of patients, which is a list of readmission indicator
    """

    # sort the dataframe by patient ID (PID) and visit date (DAY_ID)
    df = df.sort_values(['PID', 'DAY_ID'], ascending=True)

    # get important diagnoses
    diagnosis_count_df = df['DX_GROUP_DESCRIPTION'].value_counts().reset_index(name='count')
    top_diagnosis_df = diagnosis_count_df[diagnosis_count_df['count'] > diagnosis_threshold]

    # create codemap
    codemap = build_code_map(top_diagnosis_df, feature_name='index')

    # map feature to a feature id
    df['feature_id'] = df.apply(
        lambda x: codemap[x['DX_GROUP_DESCRIPTION']] if x['DX_GROUP_DESCRIPTION'] in codemap else 1, axis=1)

    # sort the dataframe by patient ID, visit date, and feature id
    df_grp = df[['PID', 'DAY_ID', 'feature_id']]
    df_grp = df_grp.drop_duplicates().sort_values(by=['PID', 'DAY_ID', 'feature_id'], ascending=True)

    # get the list of feature ids at a patient-visit level
    grouped1 = df_grp.groupby(['PID', 'DAY_ID']).agg({'feature_id': lambda x: list(x)}).reset_index()

    # get the list of visit features at a patient level
    grouped2 = grouped1.groupby(['PID']).agg({'feature_id': lambda x: list(x)}).reset_index()
    seqs = grouped2['feature_id'].tolist()

    # get inaptient hosptial events
    events = df[df['SERVICE_LOCATION'] == 'INPATIENT HOSPITAL']
    events = events[['PID', 'DAY_ID']].drop_duplicates()

    # create readmission labels for each patient visit, if a patient is readmitted within 30 days, the readmission indicator is 1
    target = (df
              .merge(events, on=['PID'], how='left')
              .assign(
        label=lambda x: np.where((x['DAY_ID_y'] >= x['DAY_ID_x']) & (x['DAY_ID_y'] < x['DAY_ID_x'] + 30), 1, 0)
    )
              .groupby(['PID', 'DAY_ID_x'], as_index=False)['label'].max()
              )
    labels = target.groupby('PID').agg({'label': lambda x: list(x)})
    labels = labels['label'].to_list()
    return seqs, labels


def train_val_test_split(seqs: list, labels: list):
    """
    split the data to train, validation, and test set

    Parameters:
    seqs: list of patients, which is a list of diagnosis descriptions at each visit date
    labels: list of patients, which is a list of readmission indicator

    Returns:
    X_train: training seqs
    Y_train: training labels
    X_val: validation seqs
    Y_val: validation labels
    X_test: test seqs
    Y_test: test labels
    """

    pickle.dump(seqs[:2000], open('data/X_train.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(labels[:2000], open('data/Y_train.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(seqs[2000:2500], open('data/X_valid.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(labels[2000:2500], open('data/Y_valid.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(seqs[2500:], open('data/X_test.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(labels[2500:], open('data/Y_test.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)


def calculate_num_features(seqs: list) -> int:
    """
    calculate the number of fetaures

    Parameters:
    seqs: list of patients (list) of visits (list) of codes (int) that contains visit sequences

    Return:
    num_features: the calculated number of features
    """
    # TODO: Calculate the number of features (diagnoses codes in the train set)
    sub_lists = [e for sublist in seqs for e in sublist]
    flat_list = [e for sublist in sub_lists for e in sublist]
    num_features = int(max(flat_list) + 1)
    return num_features



def main():
    input_path = 'data/S1_File.txt'
    df = pd.read_csv('data/S1_File.txt', sep='\t')
    seqs, labels = transform(df)
    train_val_test_split(seqs, labels)


if __name__ == '__main__':
    main()
