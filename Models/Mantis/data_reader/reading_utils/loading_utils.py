import numpy as np

from scipy.io.arff import loadarff

from .preprocessing_utils import *


def load_UEA(dataset):
    train_data = loadarff(f'/home/data3/liangzhiyu/datasets/UEA/{dataset}/{dataset}_TRAIN.arff')[0]
    test_data = loadarff(f'/home/data3/liangzhiyu/datasets/UEA/{dataset}/{dataset}_TEST.arff')[0]

    def extract_data(data):
        res_data = []
        res_labels = []
        for t_data, t_label in data:
            t_data = np.array([d.tolist() for d in t_data])
            t_label = t_label.decode("utf-8")
            res_data.append(t_data)
            res_labels.append(t_label)
        return np.array(res_data).swapaxes(1, 2), np.array(res_labels)

    train_X, train_y = extract_data(train_data)
    test_X, test_y = extract_data(test_data)

    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # scaler.fit(train_X.reshape(-1, train_X.shape[-1]))
    # train_X = scaler.transform(train_X.reshape(-1, train_X.shape[-1])).reshape(train_X.shape)
    # test_X = scaler.transform(test_X.reshape(-1, test_X.shape[-1])).reshape(test_X.shape)

    labels = np.unique(train_y)
    transform = {k: i for i, k in enumerate(labels)}
    train_y = np.vectorize(transform.get)(train_y)
    test_y = np.vectorize(transform.get)(test_y)
    return train_X, train_y, test_X, test_y


def get_label_dict(file_path):
    label_dict = {}
    with open(file_path) as file:
        lines = file.readlines()
        for line in lines:
            if '@classLabel' in line:
                label_list = line.replace('\n', '').split(' ')[2:]
                for i in range(len(label_list)):
                    label_dict[label_list[i]] = i

                break
    return label_dict


def get_data_and_label_from_ts_file(file_path, label_dict):
    with open(file_path) as file:
        lines = file.readlines()
        Start_reading_data = False
        Label_list = []
        Data_list = []
        max_length = 0
        for line in lines:
            if Start_reading_data == False:
                if '@data' in line:
                    Start_reading_data = True
            else:
                temp = line.split(':')
                Label_list.append(label_dict[temp[-1].replace('\n', '')])
                data_tuple = [np.expand_dims(np.fromstring(channel, sep=','), axis=0) for channel in temp[:-1]]
                max_channel_length = 0
                for channel_data in data_tuple:
                    if channel_data.shape[-1] > max_channel_length:
                        max_channel_length = channel_data.shape[-1]
                data_tuple = [fill_out_with_Nan(data, max_channel_length) for data in data_tuple]
                data = np.expand_dims(np.concatenate(data_tuple, axis=0), axis=0)
                Data_list.append(data)
                if max_channel_length > max_length:
                    max_length = max_channel_length

        Data_list = [fill_out_with_Nan(data, max_length) for data in Data_list]
        X = np.concatenate(Data_list, axis=0)
        Y = np.asarray(Label_list)

        return np.float32(X), Y


def multivariate_data_loader(dataset_path, dataset_name, return_train=True):
    filename_suffix = '_TRAIN.ts' if return_train else '_TEST.ts'
    dataset_path = dataset_path + '/' + dataset_name + '/' + dataset_name + filename_suffix
    label_dict = get_label_dict(dataset_path)
    X, y = get_data_and_label_from_ts_file(dataset_path, label_dict)
    return set_nan_to_zero(X), y


def construct_sliding_window_data(data, input_size, output_size, time_increment=1, feature_idx=None, target_idx=None):
    n_samples = data.shape[0] - (input_size - 1) - output_size
    range_ = np.arange(0, n_samples, time_increment)
    # handle None: select all variables
    target_idx = np.arange(data.shape[1]) if target_idx is None else target_idx
    feature_idx = np.arange(data.shape[1]) if feature_idx is None else feature_idx
    # if one value, make a list to keep dims
    target_idx = [target_idx] if type(target_idx) == int else target_idx
    feature_idx = [feature_idx] if type(feature_idx) == int else feature_idx
    x, y = list(), list()
    for i in range_:
        x.append(data[i:(i + input_size)][:, feature_idx].T)
        y.append(data[(i + input_size):(i + input_size + output_size)][:, target_idx].T)
    return np.array(x), np.array(y)


def process_forecasting_dataset_name(name):
    fragments = name.split('_')
    pred_len, feature_idx, target_idx = None, None, None
    for i, fragment in enumerate(fragments):
        if i == 0:
            dataset_name = fragment
        if 'pred=' in fragment:
            pred_len = int(fragment[5:])
        if 'feat=' in fragment:
            feats = fragment[5:].split(',')
            feature_idx = [int(feat) for feat in feats]
        if 'target=' in fragment:
            targets = fragment[7:].split(',')
            target_idx = [int(target) for target in targets]
    return dataset_name, pred_len, feature_idx, target_idx
