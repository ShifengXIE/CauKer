import pandas as pd
import numpy as np
import torch
import os
import glob

import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .reading_utils.loading_utils import *
from .reading_utils.preprocessing_utils import *


class DataReader:
    """_summary_

        Args:
            data_path: 
                Path to the folder with datasets. Defaults to '/data/'.
            transform_ts_size: 
                To which sequence length resize time series. Defaults to 512.
                If None, it doesn't transform time series and keep the original size.
            resize_func: 
                Resize function. By default, interpolate function.
            univariate: 
                If True, reshapes data from (n_samples, n_channels, seq_len) to (n_samples, n_channels * seq_len).
        """
    def __init__(self, data_path='/data/', transform_ts_size=512, resize_func=None, univariate=False):
        self.data_path = data_path
        self.transform_ts_size = transform_ts_size
        if resize_func is None:
            self.resize_func = lambda X: F.interpolate(X, size=self.transform_ts_size, mode='linear', align_corners=False)
        else:
            self.resize_func = resize_func
        self.univariate = univariate
        self._get_dataset_lists()

    def _get_dataset_lists(self,):
        self.dataset_list_ucr = os.listdir(self.data_path + "UCRArchive_2018/")
        self.dataset_list_uea = os.listdir(self.data_path + "UEA/")
        self.dataset_list_units = os.listdir(self.data_path + "UniTS/")
        self.dataset_list_others = os.listdir(self.data_path + "Others/")
        # self.dataset_list_forecasting = ['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2', 'Electricity', 'ExchangeRate', 
        #                                  'Illness', 'Traffic', 'Weather']
        self.dataset_list_forecasting = [
        'ETTh1_pred=24_feat=0', 'ETTh2_pred=24_feat=0', 
        'ETTm1_pred=96_feat=0', 'ETTm2_pred=96_feat=0',
        'Electricity_pred=24_feat=0', 'ExchangeRate_pred=30_feat=0',
        'Illness_pred=24_feat=0', 'Traffic_pred=24_feat=0', 'Weather_pred=24_feat=0'
        ]

        self.classification_datasets = self.dataset_list_ucr + self.dataset_list_uea + self.dataset_list_units + self.dataset_list_others
    
    def _get_dataset_collection(self, dataset_name):
        if dataset_name in self.dataset_list_uea:
            return "UEA/"
        elif dataset_name in self.dataset_list_units:
            return "UniTS/"
        elif dataset_name in self.dataset_list_ucr:
            return "UCRArchive_2018/"
    
    def read_dataset(self, dataset_name, which_set='train'):
        """
        dataset_name:
            For multivariate data: if the name is of the form "name:idx", then it will return only channel idx in this dataset.
        which_set:
            Either 'train', 'val' or 'test'.
        """
        if ":" in dataset_name:
            dataset_name, channel_idx = dataset_name.split(":")
            channel_idx = int(channel_idx)
        else:
            channel_idx = None

        # UCR
        if dataset_name in self.dataset_list_ucr:
            data = self._read_ucr_dataset(dataset_name, which_set=which_set)
        # UEA
        elif dataset_name in self.dataset_list_uea:
            read_func = self._read_npy_files if dataset_name == 'InsectWingbeatSubset' else self._read_ts_files
            data = read_func(dataset_name, collection='UEA', which_set=which_set, channel_idx=channel_idx)
        # Datasets from UniTS paper
        elif dataset_name in self.dataset_list_units:
            data = self._read_ts_files(dataset_name, collection='UniTS', which_set=which_set, channel_idx=channel_idx)
        # Other datasets used in pre-training
        elif dataset_name in self.dataset_list_others:
            data = self._read_other_dataset(dataset_name, which_set=which_set, channel_idx=channel_idx)
        # Forecasting datasets
        # They follow special convention:
        # The suffix "_pred={number}" specifies prediction horizon, 
        # The suffix "_feat={number1,number2,...}" specifies features to use for x
        # The suffix "_target={number1,number2,...}" specifies targets to use for y
        # To choose the sequence length for x, vary self.transform_ts_size
        # TODO: allow to choose time_increment
        elif np.any([dataset_name.startswith(ds_name) for ds_name in self.dataset_list_forecasting]):
            dataset_name_, pred_len, feature_idx, target_idx = process_forecasting_dataset_name(dataset_name)
            data = self._read_forecasting_dataset(dataset_name_, which_set=which_set, seq_len=self.transform_ts_size, pred_len=pred_len,
                                                  time_increment=1, feature_idx=feature_idx, target_idx=target_idx)
        # Some other datasets, not well investigated
        elif dataset_name == "ForecastBaseline":
            data = self._read_forecast_baseline_dataset() 
        elif dataset_name == 'SPO-PPG':
            data = self._read_spo_ppg_dataset()
        elif dataset_name == 'EEG':
            data = self._read_eeg_dataset() 
        else:
            raise KeyError('Unknown dataset name.')
        
        # encode labels to 0...K-1 if classification dataset
        x, y = data
        if dataset_name in self.classification_datasets:
            lab_encoder = LabelEncoder()
            y = lab_encoder.fit_transform(y)
        
        return x, y

    def _read_ucr_dataset(self, dataset_name, which_set):
        if which_set == 'train':
            filename_suffix = "_TRAIN.tsv"
        elif which_set == 'test':
            filename_suffix = "_TEST.tsv"
        else:
            raise KeyError('This collection has only train and test sets.')

        file_name = os.path.join(self.data_path + "UCRArchive_2018/", dataset_name, dataset_name + filename_suffix)
        data = pd.read_csv(file_name, sep='\t', header=None).to_numpy()
        X, y = torch.tensor(data[:, 1:], dtype=torch.float), data[:, 0]
        X = X.unsqueeze(-2)

        # interpolate time-series to self.transform_ts_size if not None
        if self.transform_ts_size is not None:
            X = self.resize_func(X)

        return X.numpy(), y
    
    def _read_npy_files(self, dataset_name, collection, which_set, channel_idx=None):
        if which_set == 'train':
            filename_suffix = '_train.npy'
        elif which_set == 'test':
            filename_suffix = '_test.npy'
        else:
            raise KeyError('This collection has only train and test sets.')

        base_path = os.path.join(self.data_path, collection, dataset_name)
        X, y = [np.load(base_path + '/' + s + filename_suffix) for s in ['x', 'y']]
        X = torch.tensor(X, dtype=torch.float)

        # select only this channel if specified
        if channel_idx is not None:
            X = X[:, [channel_idx], :]

        # reshape to univariate if specified
        if self.univariate:
            # flat the variable dimension
            X = X.reshape(-1, X.shape[-1])
            # insert back a variable dimension for convention
            X = X.unsqueeze(-2)
            # each entry of y is repeated the number of channels times
            y = np.repeat(y, X.shape[1])

        # interpolate time-series to self.transform_ts_size if not None
        if self.transform_ts_size is not None:
            X = self.resize_func(X)
        
        return X.numpy(), y
    

    def _read_ts_files(self, dataset_name, collection, which_set, channel_idx=None):
        if which_set == 'train':
            filename_suffix = '_TRAIN.ts'
        elif which_set == 'test':
            filename_suffix = '_TEST.ts'
        else:
            raise KeyError('This collection has only train and test sets.')

        file_name = os.path.join(self.data_path, collection, dataset_name, dataset_name + filename_suffix)

        label_dict = get_label_dict(file_name)
        X, y = get_data_and_label_from_ts_file(file_name, label_dict)
        X = set_nan_to_zero(X)
        X = torch.tensor(X, dtype=torch.float)

        # select only this channel if specified
        if channel_idx is not None:
            X = X[:, [channel_idx], :]

        # reshape to univariate if specified
        if self.univariate:            
            # Each row of X is (instance-channel) entry from original 3d matrix
            # Each entry of y is repeated the number of channels times
            y = np.repeat(y, X.shape[1])
            # flat the variable dimension
            X = X.reshape(-1, X.shape[-1])
            # insert back a variable dimension for convention
            X = X.unsqueeze(-2)

        # interpolate time-series to self.transform_ts_size if not None
        if self.transform_ts_size is not None:
            X = self.resize_func(X)
        
        return X.numpy(), y
        
    def _read_other_dataset(self, dataset_name, which_set, channel_idx=None):
        if which_set == 'train':
            filename_suffix = "train.pt"
        elif which_set == 'val':
            filename_suffix = "val.pt"
        else:
            filename_suffix = "test.pt"
        file_name = os.path.join(self.data_path + "Others/", dataset_name, filename_suffix)
        
        data = torch.load(file_name)
        X, y = data['samples'], data['labels']
        X = torch.tensor(X, dtype=torch.float)
        
        # select only this channel if specified
        if channel_idx is not None:
            X = X[:, [channel_idx], :]

        # reshape to univariate if specified
        if self.univariate:            
            # Each row of X is (instance-channel) entry from original 3d matrix
            # Each entry of y is repeated the number of channels times
            y = np.repeat(y, X.shape[1])
            # flat the variable dimension
            X = X.reshape(-1, X.shape[-1])
            # insert back a variable dimension for convention
            X = X.unsqueeze(-2)

        # interpolate time-series to self.transform_ts_size if not None
        if self.transform_ts_size is not None:
            X = self.resize_func(X)
        
        return X.numpy(), y
    
    def _read_forecasting_dataset(self, dataset_name, which_set, seq_len, pred_len, time_increment=1, feature_idx=None, target_idx=None):
        """

        """
        file_name = os.path.join(self.data_path, 'forecasting', dataset_name, dataset_name + '.csv')
        df_raw = pd.read_csv(file_name, index_col=0)

        # split train / valid / test
        n = len(df_raw)
        if 'ETTm' in dataset_name:
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif 'ETTh' in dataset_name:
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n
        
        train_df = df_raw[:train_end]
        val_df = df_raw[train_end - seq_len : val_end]
        test_df = df_raw[val_end - seq_len : test_end]

        # standardize by training set
        # according to Romain, TSMixer does not scale back, 
        # so MSE is reported on this scaled data
        scaler = StandardScaler()
        scaler.fit(train_df.values)

        train_df, val_df, test_df = [scaler.transform(df.values) for df in [train_df, val_df, test_df]]

        # train
        if which_set == 'train':
            x, y = construct_sliding_window_data(train_df, seq_len, pred_len, time_increment, feature_idx, target_idx)
        # val
        elif which_set == 'val':
            x, y = construct_sliding_window_data(val_df, seq_len, pred_len, time_increment, feature_idx, target_idx)
        # test
        else:
            x, y = construct_sliding_window_data(test_df, seq_len, pred_len, time_increment, feature_idx, target_idx)

        
        return x, y

    def _read_forecast_baseline_dataset(self):
        dataset_list=glob.glob(self.data_path+"/forecast_baseline_dataset/ALL_data/*")
        data=torch.tensor([],dtype=torch.float)
        for dataset in dataset_list[:]:
            if "txt" in dataset:
                continue
            data_file=glob.glob(dataset+"/*.csv")
            for file in data_file:
                ori_data=pd.read_csv(file)
                seg_length=ori_data.shape[0]
                while True:
                    if seg_length<64:
                        break
                    tmp=ori_data.iloc[:int(ori_data.shape[0]/seg_length)*seg_length,1:].to_numpy().T
                    tmp=tmp.reshape(-1,seg_length)
                    tmp=self.resize_func(size=(1, self.transform_ts_size),antialias=True)(torch.tensor(tmp, dtype=torch.float).unsqueeze(-2))
                    data=torch.cat((data,tmp),dim=0)
                    seg_length=int(seg_length/2)
        return data
      
    def _read_spo_ppg_dataset(self):
        dataset_list = glob.glob(self.data_path+"/ppg/osahs/*")
        data = torch.tensor([],dtype=torch.float)
        for dataset in dataset_list[:]:
            if "good" in dataset:
                continue
            tmp = torch.load(dataset).unsqueeze(dim=1)
            tmp = self.resize_func(size=(1, self.transform_ts_size),antialias=True)(torch.tensor(tmp.clone().detach(),dtype=torch.float))
            data = torch.cat((data, tmp),dim=0)
        return data
      
    def _read_eeg_dataset(self):
        file_list = ["test_data_onemodelTrue_onlyiiFalse_interTrue.npy",
                     "train_data_onemodelTrue_onlyiiFalse_interTrue.npy",
                     "train_data_icen.npy"]
        data = torch.tensor([],dtype=torch.float)
        for file in file_list[:]:
            tmp=self.resize_func(size=(1, self.transform_ts_size),antialias=True)(torch.tensor(np.load(self.data_path+"/ECG/"+file,allow_pickle=True), dtype=torch.float).unsqueeze(-2))
            data=torch.cat((data,tmp),dim=0)  
        return data
