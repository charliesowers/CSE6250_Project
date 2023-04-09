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


def get_readmission_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the readmisisn indicator for inpatient hospital stay.

    Parameters:
    df: input diagnosis dataframe

    Returns:
    readmission_label: dataframe with patient id (PID), dishcarge date (DAY_ID), and the readmission label
    """

    # sort the dataframe by patient ID (PID) and visit date (DAY_ID)
    df = df.sort_values(['PID', 'DAY_ID'], ascending=True).copy()

    # filter the dataframe for inpatient hospital stay only
    inpatient_events = df[df['SERVICE_LOCATION'] == 'INPATIENT HOSPITAL']
    inpatient_events = inpatient_events[['PID', 'DAY_ID']].drop_duplicates()

    # inpatient hospital stay can last several days, if the next hospital stay date is 1 day after the current hospital stay, then we count this as one inpatient event rather than separate events
    inpatient_events['next_inpatient_event'] = inpatient_events.groupby('PID')['DAY_ID'].transform(
        lambda x: x.shift(-1))
    inpatient_events['same_stay'] = np.where(
        (inpatient_events['next_inpatient_event'] - inpatient_events['DAY_ID']) > 1, 0, 1)

    # create the readmisison label: if the next inaptient event is within 30 days of the current inpatient event, the readmisison label is 1, otherwise 0
    inpatient_events['readmission_label'] = np.where((inpatient_events['same_stay'] == 0) & (
            (inpatient_events['next_inpatient_event'] - inpatient_events['DAY_ID']) < 30), 1, 0)

    # filter out inpatient hospital events that are belong to the same inpatient event.
    readmission_label = inpatient_events[inpatient_events['same_stay'] == 0]

    return readmission_label


def create_df(df: pd.DataFrame, diagnosis_threshold: int = 100) -> pd.DataFrame:
    """
    convert the patient-visit-diagnosis data to patient-discharge_date-readmission_label

    Parameters:
    df: input diagnosis dataframe
    diagnosis_threshold: keep diagnosis group description that occurs more than the threshold

    Returns:
    output_df: dataframe with patient id (PID), dishcarge date (DAY_ID), list of feature ids and the readmission label
    """
    # sort the dataframe by patient ID (PID) and visit date (DAY_ID)
    df = df.sort_values(['PID', 'DAY_ID'], ascending=True).copy()

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

    # get readmisison label
    readmission_label = get_readmission_label(df)

    # get final output df
    output_df = (readmission_label
                 .merge(grouped1, on=['PID'], how='left')  # merge with PID, DAY_ID, list of feature_ids
                 .loc[lambda x: x['DAY_ID_y'] <= x[
        'DAY_ID_x']]  # keep the data where the DAY_ID from grouped1 <= DAY_ID from readmisison label
                 .groupby(['PID', 'DAY_ID_x']).agg(
        {'feature_id': lambda x: list(x)}).reset_index()  # get the list of feature_ids before the inpatient event
                 .rename(columns={'DAY_ID_x': 'DAY_ID'})
                 .merge(readmission_label[['PID', 'DAY_ID', 'readmission_label']], on=['PID', 'DAY_ID'], how='inner')
                 )
    return output_df


def transform(df: pd.DataFrame, diagnosis_threshold: int = 100) -> (list, list):
    """
    split the data to seqs, labels

    Parameters:
    df: input data frame
    diagnosis_threshold: keep diagnosis group description that occurs more than the threshold

    Returns:
    seqs: list of patients (list) of visits (list) of codes (int) that contains visit sequences
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
    labels = labels['label'].tolist()

    return seqs, labels


def train_val_test_split(df: pd.DataFrame):
    """
    split the data to train, validation, and test set

    Parameters:
    df: dataframe of PID, discharge Date (DAY_ID), and readmission label
    """

    train_df = df[df['PID'] <= 2000]
    valid_df = df[(df['PID'] > 2000) & (df['PID'] <= 2500)]
    test_df = df[df['PID'] > 2500]
    pickle.dump(train_df, open('data/train.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(valid_df, open('data/valid.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    pickle.dump(test_df, open('data/test.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)


def get_seqs_labels(df: pd.DataFrame):
    """
    get list of sequence and labels from dataframe

    Parameters:
    df: dataframe of PID, Dishcarge Date (DAY_ID), and readmisison label

    Returns:
    seqs: list of patients (list) of visits (list) of codes (int) that contains visit sequences prior to the hospital discharge
    labels: list of readmisison labels
    """

    seqs = df['feature_id'].to_list()
    labels = df['readmission_label'].to_list()

    return seqs, labels


def calculate_num_features(seqs: list) -> int:
    """
    calculate the number of features

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
    df = pd.read_csv(input_path, sep='\t')
    output_df = create_df(df)
    pickle.dump(output_df, open('data/cleansed_data.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
    train_val_test_split(output_df)


if __name__ == '__main__':
    main()
