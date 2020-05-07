import numpy as np
from numpy import array
import pandas as pd

from time import time
from datetime import timedelta
from sklearn.preprocessing import StandardScaler

import os
import glob

import pyreadr

import torch

from torch.utils.data import Dataset, DataLoader

from helpers import downsample, remove_zeros, split_time_series, timer, consecutive


class BültmannUnlabledDataset(Dataset):
    
    """DataLoader for bultmann dataset without labels. the expected file type is the raw
    xlsx file"""
    
    def __init__(self, downsampling_step, sequence_length):
        loading_dataset_since = time()
        extension = 'xlsx'
        self.downsampling_step = downsampling_step
        self.sequence_length = sequence_length
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]                                            #find all files
        data_pd = pd.concat([pd.read_excel(f).iloc[2:, 4:] for f in all_filenames], ignore_index = True)   #concat all the data
        data_numpy = data_pd.to_numpy().astype(float)
        zeros_removed = remove_zeros(data_numpy)
        downsampled_data = downsample(zeros_removed, downsampling_step)
        time_series_data = split_time_series(downsampled_data, sequence_length)
        sc = StandardScaler()
        scaled_data = sc.fit_transform(time_series_data)
        scaled_data_tensor = torch.from_numpy(scaled_data)
        scaled_data_tensor_reshaped = scaled_data_tensor.unsqueeze(0).transpose(1,0)
        self.len = scaled_data_tensor_reshaped.shape[0]
        self.training_data_tensor = scaled_data_tensor_reshaped
        loading_dataset_end = time()
        hours, minutes, seconds = timer(loading_dataset_since, loading_dataset_end)
        
        print('The length of the dataset is {}'.format(len(self.training_data_tensor)))
        print("Time taken {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        
    def __getitem__(self, index):
        return self.training_data_tensor[index]
    
    def __len__(self):
        return self.len
    
    

class BültmannLabeledDataset(Dataset):
    
    """DataLoader using the labeled dataset"""
    
    def __init__(self, downsampling_step, sequence_length, train = True, normalize = False):
        loading_dataset_since = time()
        extension = 'xlsx'
        
        self.downsampling_step = downsampling_step
        self.sequence_length = sequence_length
        
        #find all files and concatenate
        all_filenames = [i for i in glob.glob('*{}'.format(extension))] 
        
        data = pd.concat([pd.read_excel(f).iloc[2:, 4:] for f in all_filenames], ignore_index = True) 
                
        #extract torque and label
        torque = data.iloc[:,0].to_numpy().astype(float)
        label = data.iloc[:,1].to_numpy().astype(float)
        
        #remove zeros from torque and label
        label = np.delete(label, np.where(torque == 0))
        torque = remove_zeros(torque)
        
        #expand dimension and store the zero removed data
        torque = np.expand_dims(torque, axis = 1)
        label = np.expand_dims(label, axis = 1)
        data = np.append(torque, label, axis = 1)
        
        #find the normal and anomalous labeled sequences and divide the data into segments'
        segmented_list = consecutive((np.where(data[:,1] == 0))[0]) + consecutive((np.where(data[:,1] == 1))[0])
        segmented_list.sort(key = lambda segmented_list: segmented_list[1])
        segmented_data = []
        for i in range(len(segmented_list)):
            segments = segmented_list[i]
            start_index = segments[0]
            end_index = segments[len(segments)-1]
            seg_data = data[start_index:end_index+1, :]
            segmented_data.append(seg_data)
            
        #downsample the data and make sequences'
        sequenced_data = []
        for i in range(len(segmented_data)):
            label = segmented_data[i][0,1]
            data = downsample(segmented_data[i][:,0], self.downsampling_step)
            data = split_time_series(data, self.sequence_length)
            if label == 0.:
                label_column = [0]*len(data)
            else:
                label_column = [1]*len(data)
                
            sequenced_data.append(np.column_stack((data, label_column)))
            
        data = np.empty((0,self.sequence_length+1))
            
        for i in range(len(sequenced_data)):
            if sequenced_data[i].shape[1] == self.sequence_length+1:
                data = np.append(data, sequenced_data[i], axis = 0)
                
        if normalize:
            #scale the data and return the tensor output'
            sc = StandardScaler()
        
            training_data = data[0:int(0.7*(len(data))),0:self.sequence_length]
            testing_data = data[int(0.7*(len(data))):,0:self.sequence_length]
        
            training_label = data[0:int(0.7*(len(data))),-1]
            testing_label = data[int(0.7*(len(data))):,-1]
        
            sc_fit = sc.fit(training_data)
        
            if train:
                unlabeled_data = sc_fit.transform(training_data)
                data = np.column_stack((unlabeled_data, training_label))
            else:
                unlabeled_data = sc_fit.transform(testing_data)
                data = np.column_stack((unlabeled_data, testing_label))
        else:
            if train:
                data = data[0:int(0.7*(len(data))),:]
            else:
                data = data[int(0.7*(len(data))):,:]
        data = torch.from_numpy(data).unsqueeze(0).transpose(1,0)
        
        self.len = data.shape[0]
        self.data = data
        
        loading_dataset_end = time()
        hours, minutes, seconds = timer(loading_dataset_since, loading_dataset_end)
        
        print('The length of the dataset is {}'.format(self.len))
        print("Time taken {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        
    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return self.len 


class TimeSeriesTEastmanFull(Dataset):
    
    """TEastman full dataset dataloader. expected file of RData type"""
    
    def __init__(self, sequence_length, downsampling_step = 10, train = True, normalize = True):
        
        self.sequence_length = sequence_length
        self.downsampling_step = downsampling_step
        self.train = train
        self.normalize = normalize
        
        #load the data
        if self.train:
            #fault free training data
            
            load = pyreadr.read_r('.\TE_Data_full\TEP_FaultFree_Training.RData')
            load = load['fault_free_training']
            temp_data = np.asarray(load, dtype=np.float32)[ :,3:]
            #temp_label = np.asarray(load, dtype=np.int32)[:,0]
            
            if self.normalize:
                sc = StandardScaler()
                temp_data = sc.fit_transform(temp_data)
                
            fault_free_training = temp_data
            del(temp_data)
            
            #make sequences
            sequenced_data_list = []
            
            for variables in fault_free_training.T:
                temp_data_d = downsample(variables, self.downsampling_step)
                temp_data = split_time_series(temp_data_d, self.sequence_length)
                temp_data = np.expand_dims(temp_data, axis = 1)
                sequenced_data_list.append(temp_data)
            
            fault_free_training = torch.empty(((sequenced_data_list[0].shape[0]), 0, self.sequence_length))
            
            for sequences in sequenced_data_list:
                fault_free_training = torch.cat((fault_free_training, torch.from_numpy(sequences)), dim = 1)
                
            fault_free_label = np.zeros(fault_free_training.shape[0], dtype = np.int32)
            
            
                
            del(sequenced_data_list)
            del(temp_data)
            
            #faulty training data
            
            load = pyreadr.read_r('.\TE_Data_full\TEP_Faulty_Training.RData') 
            load = load['faulty_training']
            temp_data = np.asarray(load, dtype=np.float32)[ :,3:]
           
            temp_label = np.asarray(load, dtype=np.int32)[:,0]
            
            if self.normalize:
                temp_data = sc.fit_transform(temp_data)
                
            faulty_training = temp_data
            del(temp_data)            
            
            #make sequences
            sequenced_data_list = []
            
            for variables in faulty_training.T:
                temp_data_d = downsample(variables, self.downsampling_step)
                temp_data = split_time_series(temp_data_d, self.sequence_length)
                temp_data = np.expand_dims(temp_data, axis = 1)
                sequenced_data_list.append(temp_data)
                
            faulty_training = torch.empty((sequenced_data_list[0].shape[0], 0, self.sequence_length))
            
            for sequences in sequenced_data_list:
                
                faulty_training = torch.cat((faulty_training, torch.from_numpy(sequences)), dim = 1)
                
            faulty_label = np.ones(faulty_training.shape[0], dtype = np.int32)

            del(sequenced_data_list)
            del(temp_data)
            
            training_data = torch.cat((fault_free_training, faulty_training), dim = 0)
            training_label = np.concatenate((fault_free_label, faulty_label), axis = 0)
            
            self.data = training_data
            self.label = torch.from_numpy(training_label)
            self.len = len(self.data)
            print('The length of the dataset is {}'.format(self.len))
            
        else:
            
            #fault free testing data
            
            load = pyreadr.read_r('.\TE_Data_full\TEP_FaultFree_Testing.RData')
            load = load['fault_free_testing']
            temp_data = np.asarray(load, dtype=np.float32)[ :,3:]
            #temp_label = np.asarray(load, dtype=np.int32)[:,0]
            
            if self.normalize:
                sc = StandardScaler()
                temp_data = sc.fit_transform(temp_data)
                
            fault_free_testing = temp_data
            del(temp_data)
            
            #make sequences
            sequenced_data_list = []
            
            for variables in fault_free_testing.T:
                temp_data_d = downsample(variables, self.downsampling_step)
                temp_data = split_time_series(temp_data_d, self.sequence_length)
                temp_data = np.expand_dims(temp_data, axis = 1)
                sequenced_data_list.append(temp_data)
            
            fault_free_testing = torch.empty(((sequenced_data_list[0].shape[0]), 0, self.sequence_length))
            
            for sequences in sequenced_data_list:
                fault_free_testing = torch.cat((fault_free_testing, torch.from_numpy(sequences)), dim = 1)
                
            fault_free_label = np.zeros(fault_free_testing.shape[0], dtype = np.int32)
            
            
                
            del(sequenced_data_list)
            del(temp_data)
            
            #faulty testing data
            
            load = pyreadr.read_r('.\TE_Data_full\TEP_Faulty_Testing.RData') 
            load = load['faulty_testing']
            temp_data = np.asarray(load, dtype=np.float32)[ :,3:]
           
            #temp_label = np.asarray(load, dtype=np.int32)[:,0]
            
            if self.normalize:
                sc = StandardScaler()
                temp_data = sc.fit_transform(temp_data)
                
            faulty_testing = temp_data
            del(temp_data)            
            
            #make sequences
            sequenced_data_list = []
            
            for variables in faulty_testing.T:
                temp_data_d = downsample(variables, self.downsampling_step)
                temp_data = split_time_series(temp_data_d, self.sequence_length)
                temp_data = np.expand_dims(temp_data, axis = 1)
                sequenced_data_list.append(temp_data)
                
            faulty_testing = torch.empty((sequenced_data_list[0].shape[0], 0, self.sequence_length))
            
            for sequences in sequenced_data_list:
                
                faulty_testing = torch.cat((faulty_testing, torch.from_numpy(sequences)), dim = 1)
                
            faulty_label = np.ones(faulty_testing.shape[0], dtype = np.int32)

            del(sequenced_data_list)
            del(temp_data)
            
            testing_data = torch.cat((fault_free_testing, faulty_testing), dim = 0)
            testing_label = np.concatenate((fault_free_label, faulty_label), axis = 0)
            
            self.data = testing_data
            self.label = torch.from_numpy(testing_label)
            self.len = len(self.data)
            print('The length of the dataset is {}'.format(self.len))
            
            
    def __getitem__(self, index):
        
        return self.data[index], self.label[index]
    
    def __len__(self):
        
        return self.len
         
        
class TEastmanDataset(torch.utils.data.Dataset):
    
    """Default Tenneessee Eastman DataLoader """
    
    def __init__(self, folder="./TE_process/", train=True, cases=range(22), normalize=False, onehot=False):
        self.DATA=torch.zeros(0, 52)
        self.LABEL=torch.zeros(0)
        if train:
            ending=".dat"
        else:
            ending="_te.dat"

        for error_mode in cases:
            if error_mode<10:
                name=folder+"d0"+str(error_mode)+ending
            else:
                name=folder+"d"+str(error_mode)+ending

            data=torch.tensor(np.genfromtxt(name)).float()

            if error_mode==0 and train:
                data=data.permute(1,0)

            labels=torch.tensor(error_mode).view(1).float().repeat(data.shape[0])

            self.DATA=torch.cat((self.DATA, data), dim=0)
            self.LABEL=torch.cat((self.LABEL, labels), dim=0)

        if normalize:
            # self.DATA = (self.DATA - self.DATA.mean(dim=0))/self.DATA.std(dim=0)
            mins,_ = self.DATA.min(dim=0)
            maxs,_ = self.DATA.max(dim=0)
            self.DATA = (self.DATA - mins)/(maxs - mins)

        if onehot:
            helper=torch.zeros(self.LABEL.shape[0], 2)

            for batch in range(self.LABEL.shape[0]):
                helper[batch, int(self.LABEL[batch])]=1
            self.LABEL=helper
        self.LABEL=self.LABEL.long()

    def __len__(self):
        return self.DATA.shape[0]

    def __getitem__(self, idx):
        return self.DATA[idx], self.LABEL[idx]
        

            
class TimeSeriesTEastmanDataset(TEastmanDataset):
    
    """Tenneessee Eastman Time Series DataLoader"""
    
    def __init__(self, folder="./TE_process/", sequence_length=20, stride=10,
                 train=True, cases=range(22), normalize=False, onehot=False):
        super(TimeSeriesTEastmanDataset, self).__init__(folder=folder, train=train, cases=cases, normalize=normalize, onehot=onehot)
        
        self.SEQUENCES , self.LABELS = self._getSequences(self.DATA, self.LABEL, cases, sequence_length, stride)
        
        self.LABEL_MAPPER = {}
        
        for idx, case in enumerate(cases):
            self.LABEL_MAPPER[case] = idx
        
    def __len__(self):
        return self.SEQUENCES.shape[0]

    def __getitem__(self, idx):
        label = self.LABEL_MAPPER[self.LABELS[idx][0].item()]
        return self.SEQUENCES[idx], label
    
    @staticmethod
    def _getSequences(data, label, cases, sequence_length, stride):
        sequenceLists = []
        labelLists = []
        for case in cases:
            idx = label == case
            data_subset = data[idx]
            label_subset = label[idx]        
            subset_length = data_subset.shape[0]
            num_seq = int(subset_length - sequence_length + stride)
            for x in range(0, num_seq, stride):
                if x+sequence_length < data_subset.shape[0]:
                    
                    sequenceLists.append(data_subset[x:x+sequence_length])
                    labelLists.append(label_subset[x:x+sequence_length])
        # Permute last two dimensions to convert data into [numChannels, sequenceLength]
                    # to be consistent with 1d Convolutional layers of PyTorch
        sequenceLists = torch.stack(sequenceLists).permute(0, 2, 1)
        print(sequenceLists.shape)
        labelLists = torch.stack(labelLists)
        return sequenceLists, labelLists
     
    
def loadTimeSeriesTEData(minibatch=60, sequence_length=20, stride=1, cases=range(22), normalize= False):
    
    train_loader = torch.utils.data.DataLoader(
            TimeSeriesTEastmanDataset(sequence_length=sequence_length, stride=stride, train=True, cases=cases, normalize=normalize),
                                               batch_size=minibatch, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
            TimeSeriesTEastmanDataset(sequence_length=sequence_length, stride=stride, train=False, cases=cases, normalize=normalize),
                                               batch_size=minibatch, shuffle=True, pin_memory=True)
    
    return list(enumerate(train_loader)), list(enumerate(test_loader)), train_loader, test_loader, cases


 
    
    
   

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    