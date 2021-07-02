import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn.naive_bayes


def load_data(path: str) -> pd.DataFrame:
    """
    Loads the provided data and returns it as a DataFrame

    :param path: path to the provided DataSet
    :return: DataFrame with provided Data
    """
    f = pd.read_csv(path, sep=',')
    f.columns = f.columns.str.replace('#->', '')
    f['Timestamp'] = pd.to_datetime((f['Timestamp'] * 1000).astype(int), unit='ms')
    return f.set_index('Timestamp')


def resample(data: pd.DataFrame, timedelta: str) -> pd.DataFrame:
    """
    Resamples the data to the provided sample rate

    :param data: DataFrame which contains the loaded dataset
    :param timedelta: timedelta the dataset should have, eg.: '1s', '10ms' ...
    :return: DataFrame with resampled data
    """

    # maybe first split by label and resample after
    label = data['label'].resample(timedelta).first().fillna(method='ffill')
    data = data.resample(timedelta).mean().fillna(method='ffill')
    data['label'] = label
    return data


def sliding_windows(data: pd.DataFrame, window_size: int, overlap: float) -> list:
    """
    Takes the data and split it into overlapping windows

    :param data: DataFrame which contains the loaded dataset
    :param window_size: size the sliding window should have
    :param overlap: percentage of overlap (between 0 and 1)
    :return: list of windows
    """
    overlap_size = int(window_size*overlap)
    window_start = np.arange(len(data))[::overlap_size]
    window_frame = list(zip(window_start, window_start + window_size))
    return [data.iloc[w[0]:w[1]] for w in window_frame]


def group_label(data: pd.DataFrame) -> dict:
    """
    Groups the DataFrame by label and returns a dictionary with the labels as key and a list of DataFrames
    with all occurrences of that label

    :param data: DataFrame which contains the loaded dataset
    :return: dictionary with labeled Data
    """
    data['group'] = data['label'].ne(data['label'].shift()).cumsum()
    group = data.groupby('group')
    label_dict = {}
    for name, data in group:
        label = data.iloc[0]['label']
        if label not in label_dict.keys():
            label_dict[label] = []
        label_dict[label].append(data.drop(columns=['group']))
    return label_dict


def filter_device(data: pd.DataFrame, dev: str, sensor: str = None) -> pd.DataFrame:
    """
    Filters the already grouped data (eg all 'sitting' data) for devices and their sensors

    :param data: grouped data
    :param dev: device you want to filter
    :param sensor: sensor data you want to filter (optional, if you want to have sensor data of all devices,
                    leave it empty)
    :return: filtered data
    """
    if sensor:
        dev_cols = [col for col in data.columns if dev in col and sensor in col]
    else:
        dev_cols = [col for col in data.columns if dev in col]

    dev_cols.append('label')

    return data[dev_cols]


if __name__ == '__main__':
    person_file = load_data(r"D:\Uni\dataset\p2_phone_watch_sensorFusion.ctm")
    person_file = resample(person_file, '1s')
    label_dictionary = group_label(person_file)
    filter_device(label_dictionary['downstairs'][0], 'phone', 'ACCEL').plot()
    filter_device(label_dictionary['downstairs'][0], 'watch', 'ACCEL').plot()
    plt.show()

    windows = sliding_windows(person_file, 200, 0.2)

