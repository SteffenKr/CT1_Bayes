import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_fscore_support as score
from prettytable import PrettyTable
import os
import glob
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_folder(path: str, converted: bool = False) -> list:
    """
    Loads all files in a Folder

    :param path: folder Path
    :param converted: are the files original or converted
    :return:
    """
    if converted:
        return [load_converted_data(os.path.join(path, filename))
                for filename in glob.glob(os.path.join(path, '*.csv'))]
    else:
        return [load_data(os.path.join(path, filename))
                for filename in glob.glob(os.path.join(path, '*.ctm'))]


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


def load_converted_data(path: str) -> pd.DataFrame:
    """
    Loads the "converted" provided data and returns it as a DataFrame

    :param path: path to the provided DataSet
    :return: DataFrame with provided Data
    """
    f = pd.read_csv(path, sep=',')
    f.columns = f.columns.str.replace('#->', '')
    f.columns = f.columns.str.replace('"', '')
    f['watch_TYPE_GYROSCOPE-Z'] = f['watch_TYPE_GYROSCOPE-Z'].str[:-1]
    f['watch_TYPE_GYROSCOPE-Z'] = f['watch_TYPE_GYROSCOPE-Z'].astype(float)
    f['Timestamp'] = f['Timestamp'].str[1:]
    f['Timestamp'] = pd.to_datetime((f['Timestamp'].astype(float) * 1000).astype(int), unit='ms')
    return f.set_index('Timestamp')


def resample(data: pd.DataFrame, timedelta: str) -> pd.DataFrame:
    """
    Resamples the data to the provided sample rate

    :param data: DataFrame which contains the loaded dataset
    :param timedelta: timedelta the dataset should have, eg.: '1s', '10ms' ...
    :return: DataFrame with resampled data
    """

    print(".", end='')
    label = data['label'].resample(timedelta).first().fillna(method='ffill')
    data = data.resample(timedelta).mean().fillna(method='ffill')
    data['label'] = label
    return data


def sliding_windows(data: pd.DataFrame, w_size: int, overlap: float) -> (list, np.array):
    """
    Takes the data and split it into overlapping windows

    :param data: DataFrame which contains the loaded dataset
    :param w_size: size the sliding window should have
    :param overlap: percentage of overlap (between 0 and 1)
    :return: list of windows and their labels
    """
    overlap_size = int(w_size * overlap)
    window_start = np.arange(len(data))[::w_size - overlap_size]
    window_frame = list(zip(window_start, window_start + w_size))

    windows = [data.iloc[w[0]:w[1]] for w in window_frame]
    window_labels = np.array([w['label'].value_counts().index[0] for w in windows])
    return windows, window_labels


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


def extract_features(data: pd.DataFrame, used_features: dict) -> list:
    """
    Extracts features from Data

    :param data:
    :param used_features:
    :return:
    """
    features = []
    if used_features['mean']:
        features.append(data.mean().values.tolist())
    if used_features['std']:
        features.append(data.std().values.tolist())
    if used_features['var']:
        features.append(data.var().values.tolist())
    if used_features['min']:
        features.append(data.min(numeric_only=True).values.tolist())
    if used_features['max']:
        features.append(data.max(numeric_only=True).values.tolist())

    return [j for sub in features for j in sub]


def extract_file_features(data: list, w_size: int, overlap_percentage: float, used_features: dict) -> (list, list):
    """
    Extracts features from Data Set

    :param data: provided Data set
    :param w_size: size of sliding window
    :param overlap_percentage: overlap percentage 0 = 0%. 1 = 100%
    :param used_features: dict which tells which features should be used
    :return:
    """
    x = []
    y = []
    print("Extracting features: ", end='')
    for person in data:
        print(".", end='')
        wind, labels = sliding_windows(person, w_size, overlap_percentage)
        x.append([extract_features(w, used_features) for w in wind])
        y.append(labels)
    print()
    return x, y


def print_score(labels: list, predictions: list) -> None:
    """
    Takes the actual labels and the predictions of the cross validation and calculates the mean score

    :param labels: actual labels
    :param predictions: prediction from cross validation
    :return:
    """
    action_labels = ['sitting', 'standing', 'reading', 'typing', 'hand_wash',
                     'dishes', 'vacuum', 'walking', 'downstairs', 'upstairs']
    all_precision = []
    all_recall = []
    all_fscore = []

    for n in range(len(labels)):
        precision, recall, fscore, _ = score(labels[n], predictions[n], labels=action_labels)
        all_precision.append(precision)
        all_recall.append(recall)
        all_fscore.append(fscore)

    mean_precision = np.mean(all_precision, axis=0)
    mean_recall = np.mean(all_recall, axis=0)
    mean_fscore = np.mean(all_fscore, axis=0)

    t = PrettyTable()
    t.add_column("labels", action_labels)
    t.add_column("precision", [float('%.3g' % n) for n in mean_precision])
    t.add_column("recall", [float('%.3g' % n) for n in mean_recall])
    t.add_column("fscore", [float('%.3g' % n) for n in mean_fscore])
    print(t)


def split_data(x_data: list, y_data: list) -> (list, list):
    """
    splits the data for cross validation

    :param x_data: features
    :param y_data: labels
    :return:
    """
    print("Splitting data: ", end='')

    v_data = []
    t_data = []
    for n in range(len(x_data)):
        print(".", end='')
        train_x = x_data[:n] + x_data[n+1:]
        train_y = y_data[:n] + y_data[n+1:]
        t_data.append({'x': [item for sublist in train_x for item in sublist],
                       'y': [item for sublist in train_y for item in sublist]})

        v_data.append({'x': x_data[n],
                       'y': y_data[n]})

    print()

    return t_data, v_data


def ten_fold(t_data: list, v_data: list):
    """
    Does 10-Fold cross Validation

    :param t_data: training Data
    :param v_data: validation Data
    :return:
    """
    all_predictions = []
    all_true_labels = []
    for n in range(10):
        gaus = GaussianNB()
        gaus.fit(t_data[n]['x'], t_data[n]['y'])
        all_predictions.append(gaus.predict(v_data[n]['x']))
        all_true_labels.append(v_data[n]['y'])
    print_score(all_true_labels, all_predictions)


if __name__ == '__main__':
    ''' Input Parameter '''
    step_size = '100ms'
    window_size = 100
    window_overlap = 0.5
    folder_path = r"C:\Uni\ct1"
    converted_files = True
    use_features = {'mean': True,
                    'std': True,
                    'var': False,
                    'min': True,
                    'max': True}

    ''' Loading Data'''
    files = load_folder(folder_path, converted=converted_files)

    ''' Resampling Data '''
    print("Resampling: ", end='')
    files = [resample(file, step_size) for file in files]
    print()

    ''' Extracting Features'''
    x_files, y_files = extract_file_features(files, window_size, window_overlap, use_features)

    ''' Split Data for 10-fold'''
    training_data, validation_data = split_data(x_files, y_files)

    ''' Ten Fold Validation'''
    ten_fold(training_data, validation_data)
