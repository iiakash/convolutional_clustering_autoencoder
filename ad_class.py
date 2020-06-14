import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import functions
import networks
import dataset

import sklearn
import os

import matplotlib.pyplot as plt
import seaborn as sns
import time

class AnomalyDetection():
    
    def __init__(self, mode, network, network_criterion, n_clusters, clustering_criterion, cluster_update_interval,
                 no_of_clustering_channels, n_epochs, no_of_pretrain_epochs, batch_size, lr, alpha,
                 downsampling_step, sequence_length, kernel_size):
        self.mode = mode
        self.network = network
        self.network_criterion = network_criterion
        self.n_clusters = n_clusters
        self.clustering_criterion = clustering_criterion
        self.cluster_update_interval = cluster_update_interval
        self.no_of_clustering_channels = no_of_clustering_channels
        self.n_epochs = n_epochs
        self.no_of_pretrain_epochs = no_of_pretrain_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.alpha = alpha
        self.downsampling_step = downsampling_step
        self.sequence_length = sequence_length
        self.kernel_size = kernel_size
        
        
        self.path = './{} N{} NC{} Ncl{} CUI{} NclCh{} E{} PE{} BS{} LR{} A{} DS{} SL{} KS{}'.format(self.mode, self.network.__class__.__name__,
                         self.network_criterion, self.n_clusters,
                self.cluster_update_interval, self.no_of_clustering_channels,
                self.n_epochs, self.no_of_pretrain_epochs, self.batch_size, self.lr, self.alpha, self.downsampling_step,self.sequence_length, self.kernel_size)
        
        functions.createFolder(self.path)
        
    def load_bultmann_data(self, normalize = True):
        
        tr_dataset = dataset.BültmannLabeledDataset(self.downsampling_step, self.sequence_length, 
                                                    train = True, normalize = normalize)
        tr_loader = DataLoader(tr_dataset, batch_size = self.batch_size, shuffle = False)
        
        te_dataset = dataset.BültmannLabeledDataset(self.downsampling_step, self.sequence_length, 
                                                    train = False, normalize = normalize)
        te_loader = DataLoader(te_dataset, batch_size = self.batch_size, shuffle = False)
        
        
        return tr_loader, len(tr_dataset), te_loader, len(te_dataset)
    
    def load_teastman_data(self, normalize = True):
        
        _, _, tr_loader, te_loader, cases = dataset.loadTimeSeriesTEData(minibatch = self.batch_size, sequence_length = self.sequence_length, 
                                                    stride = 1, cases = range(22), normalize = normalize)
        
        
        return tr_loader, te_loader, cases
        
    def load_teastman_full_data(self, train = True, normalize = True):
        
        data_set = dataset.TimeSeriesTEastmanFull(sequence_length = self.sequence_length, downsampling_step = self.downsampling_step, train = train, normalize = normalize)
        
        dataloader = DataLoader(data_set, batch_size = self.batch_size, shuffle = False)
        
        return dataset, dataloader
        
    
    def train_bultmann(self, tr_loader, tr_dataset_length, Adam = True, scheduler = True):
        
        since = time.time()
        print('Training the network {}'.format(self.network.__class__.__name__))
        print('Network Architecture \n{}'.format(self.network))
        print('Network Criterion {}'.format(self.network_criterion))
        list_of_network_loss = []
        list_of_clustering_loss = []
        list_of_total_loss = []
        list_of_losses = []
        learning_rates = []
        list_of_centers = []
        list_of_ranks_of_center_distances = []
        list_of_center_distances = []
        
        if Adam:
            optimizer = torch.optim.Adam(self.network.parameters(), lr = self.lr, weight_decay = 0.0)
            
        else:
            optimizer = torch.optim.SGD(self.network.parameters(), lr = self.lr, momentum = 0.0, weight_decay = 0.0, nesterov = False)
            
        if scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.1)
        
        for epoch in range(self.n_epochs):
        
            embedded_representation = []
            batched_center_index = 0                                                
            total_combined_loss = 0.0
            total_network_loss = 0.0
            total_clustering_loss = 0.0
            labels = np.empty((0,1), float)
                
            for batch in tr_loader:
            
                #extract the sequence and label from the batch and make predictions and return bottleneck
            
                sequences = batch[:,:,0:self.sequence_length].float()                                     
                batch_labels = batch[:,:,self.sequence_length]
                labels = np.append(labels, batch_labels.numpy(), axis = 0)
                target_sequences = sequences.clone()
                predictions, bottleneck = self.network(sequences)
                
                embedded_representation.append(bottleneck.clone().detach())
                batch_embeddings = torch.cat(embedded_representation) 
                
                #compute the network loss
            
                network_loss = self.network_criterion(predictions, target_sequences)
            
                #set condition for pretrain mode
            
                if epoch <= self.no_of_pretrain_epochs:
                
                    #pretrain mode
                    
                    clustering_loss = torch.zeros([1,1], dtype = torch.float64)
                    combined_loss = network_loss      # + self.alpha*clustering_loss   # defining the combined loss
                    optimizer.zero_grad()
                
                    #calculating the gradients and taking step with only network loss as the clustering loss is zero'
                
                    combined_loss.backward(retain_graph = True)                     # retaining the pytorch computation graph so that backward can be done twice
                    optimizer.step()
                    
                    
                    
                else:
                
                    #joint training mode
                
                    clustering_loss = self.clustering_criterion(bottleneck, batched_center_designation[batched_center_index])
                    batched_center_index += 1                                       # incrementing the batched center index
                    combined_loss = (1- self.alpha)*network_loss + self.alpha*clustering_loss
                    optimizer.zero_grad()
                
                    #calculating the gradients but not taking step
                
                    combined_loss.backward(retain_graph = True)
                
                    #updating the weights of the clustering friendly channels wrt combined loss
                
                    bottleneck_layer = functions.get_bottleneck_name(self.network)
            
                    #train_reporter.print_grads(network)
                
                    with torch.no_grad():
                    
                        for name, parameters in self.network.named_parameters():
                        
                            if name == bottleneck_layer:
                            
                                ranked_channels = torch.from_numpy(ranks_of_center_distances)
                                parameters.grad[torch.where(ranked_channels <= self.no_of_clustering_channels)] = 0.0
                            
                    optimizer.step()
        
                    #updating the weights of rest of the channels wrt network loss'
                
                    optimizer.zero_grad()
                    network_loss.backward()
                
                    with torch.no_grad():
                    
                        for name, parameters in self.network.named_parameters():
                        
                            if name == bottleneck_layer:
                            
                                ranked_channels = torch.from_numpy(ranks_of_center_distances)
                                parameters.grad[torch.where(ranked_channels > self.no_of_clustering_channels)] = 0.0
                            
                    optimizer.step()
                    
                    
                    
                total_network_loss += network_loss.item()
                total_clustering_loss += clustering_loss.item()
                total_combined_loss += combined_loss.item()
            #extract embeddings
            embeddings = batch_embeddings
        
            #make list of losses
        
            list_of_network_loss.append(total_network_loss/(tr_dataset_length)/self.batch_size)
            list_of_clustering_loss.append(total_clustering_loss/(tr_dataset_length)/self.batch_size)
            list_of_total_loss.append(total_combined_loss/(tr_dataset_length)/self.batch_size)
        
            #make cluster update interval array
        
            cluster_update = np.arange(self.no_of_pretrain_epochs, self.n_epochs, self.cluster_update_interval)
        
            #clustering
            for update in cluster_update:
            
                if update == epoch:
                    print('Updating Cluster Centers')
                    center_designation_pre = []
                    cluster_label_pre = []
                    centers_pre = []
                    no_of_channels = embeddings.shape[1]
                
                    for i in range(no_of_channels):
                        channel = embeddings[:,i,:].numpy()
                        choice_cluster, initial_centers, cluster_ass = functions.kmeansalter(channel, self.n_clusters)
                        cluster_label_pre.append(torch.from_numpy(choice_cluster).unsqueeze(0).transpose(1,0))
                        cluster_label = torch.cat(cluster_label_pre, dim = 1)
                        centers_pre.append(torch.from_numpy(initial_centers).unsqueeze(0).transpose(1,0))
                        centers = torch.cat(centers_pre, dim = 1)
                        center_designation_pre.append(cluster_ass.unsqueeze(0).transpose(1,0))
                        center_designation = torch.cat(center_designation_pre, dim = 1)
                        
                    
                    batched_center_designation = list(functions.divide_batches(center_designation, self.batch_size))
                    center_distances, ranks_of_center_distances = functions.rank_channels(centers)
                    
                
            print('Epoch : {}/{} Network Loss : {} Clustering Loss : {} Total Loss : {}'.format(epoch+1, 
              self.n_epochs, (total_network_loss/(tr_dataset_length/self.batch_size)), 
          (total_clustering_loss/(tr_dataset_length/self.batch_size)),
          (total_combined_loss/(tr_dataset_length/self.batch_size))))
        
        list_of_centers.append(centers.numpy())
        list_of_ranks_of_center_distances.append(ranks_of_center_distances)
        list_of_center_distances.append(center_distances)
        list_of_losses.append(list_of_network_loss)
        list_of_losses.append(list_of_clustering_loss)
        list_of_losses.append(list_of_total_loss)
        end = time.time()
        hours, minutes, seconds = functions.timer(since, end)
        print("Time taken {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
        return self.network, optimizer, list_of_network_loss, list_of_clustering_loss, list_of_total_loss, list_of_losses, embeddings, labels, list_of_centers, list_of_ranks_of_center_distances, list_of_center_distances
    
    def train_teastman(self, tr_loader, tr_dataset_length, Adam = True, scheduler = True):
        
        since = time.time()
        print('Training the network {}'.format(self.network.__class__.__name__))
        print('Network Architecture \n{}'.format(self.network))
        print('Network Criterion {}'.format(self.network_criterion))
        list_of_network_loss = []
        list_of_clustering_loss = []
        list_of_total_loss = []
        list_of_losses = []
        learning_rates = []
        list_of_centers = []
        list_of_ranks_of_center_distances = []
        list_of_center_distances = []
        if Adam:
            optimizer = torch.optim.Adam(self.network.parameters(), lr = self.lr, weight_decay = 0.3)
            
        else:
            optimizer = torch.optim.SGD(self.network.parameters(), lr = self.lr, momentum = 0.2, weight_decay = 0.0, nesterov = True)
            
        if scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.1)
        
        for epoch in range(self.n_epochs):
        
            embedded_representation = []
            batched_center_index = 0                                                
            total_combined_loss = 0.0
            total_network_loss = 0.0
            total_clustering_loss = 0.0
            labels = np.empty((1), int)
                
            for data, label in tr_loader:
            
                #extract the sequence and label from the batch and make predictions and return bottleneck
            
                sequences = data                                    
                batch_labels = label
                labels = np.append(labels, batch_labels.numpy(), axis = 0)
                target_sequences = sequences.clone()
                predictions, bottleneck = self.network(sequences)
                embedded_representation.append(bottleneck.clone().detach())
                batch_embeddings = torch.cat(embedded_representation) 
                
                #compute the network loss
            
                network_loss = self.network_criterion(predictions, target_sequences)
            
                #set condition for pretrain mode
            
                if epoch <= self.no_of_pretrain_epochs:
                
                    #pretrain mode
                    
                    clustering_loss = torch.zeros([1,1], dtype = torch.float64)
                    combined_loss = network_loss      # + self.alpha*clustering_loss   # defining the combined loss
                    optimizer.zero_grad()
                
                    #calculating the gradients and taking step with only network loss as the clustering loss is zero'
                
                    combined_loss.backward(retain_graph = True)                     # retaining the pytorch computation graph so that backward can be done twice
                    optimizer.step()
                    
                    
                    
                else:
                
                    #joint training mode
                
                    clustering_loss = self.clustering_criterion(bottleneck, batched_center_designation[batched_center_index])
                    batched_center_index += 1                                       # incrementing the batched center index
                    combined_loss = (1- self.alpha)*network_loss + self.alpha*clustering_loss
                    optimizer.zero_grad()
                
                    #calculating the gradients but not taking step
                
                    combined_loss.backward(retain_graph = True)
                
                    #updating the weights of the clustering friendly channels wrt combined loss
                
                    bottleneck_layer = functionsfunctions.get_bottleneck_name(self.network)
            
                    #train_reporter.print_grads(network)
                
                    with torch.no_grad():
                    
                        for name, parameters in self.network.named_parameters():
                        
                            if name == bottleneck_layer:
                            
                                ranked_channels = torch.from_numpy(ranks_of_center_distances)
                                parameters.grad[torch.where(ranked_channels <= self.no_of_clustering_channels)] = 0.0
                            
                    optimizer.step()
        
                    #updating the weights of rest of the channels wrt network loss'
                
                    optimizer.zero_grad()
                    network_loss.backward()
                
                    with torch.no_grad():
                    
                        for name, parameters in self.network.named_parameters():
                        
                            if name == bottleneck_layer:
                            
                                ranked_channels = torch.from_numpy(ranks_of_center_distances)
                                parameters.grad[torch.where(ranked_channels > self.no_of_clustering_channels)] = 0.0
                            
                    optimizer.step()
                    
                    
                    
                total_network_loss += network_loss.item()
                total_clustering_loss += clustering_loss.item()
                total_combined_loss += combined_loss.item()
            #extract embeddings
            embeddings = batch_embeddings
        
            #make list of losses
        
            list_of_network_loss.append(total_network_loss/(tr_dataset_length)/self.batch_size)
            list_of_clustering_loss.append(total_clustering_loss/(tr_dataset_length)/self.batch_size)
            list_of_total_loss.append(total_combined_loss/(tr_dataset_length)/self.batch_size)
        
            #make cluster update interval array
        
            cluster_update = np.arange(self.no_of_pretrain_epochs, self.n_epochs, self.cluster_update_interval)
        
            #clustering
            for update in cluster_update:
            
                if update == epoch:
                    since = time.time()
                    print('Updating Cluster Centers')
                    center_designation_pre = []
                    cluster_label_pre = []
                    centers_pre = []
                    no_of_channels = embeddings.shape[1]
                
                    for i in range(no_of_channels):
                        channel = embeddings[:,i,:].numpy()
                        choice_cluster, initial_centers, cluster_ass = functions.kmeansalter(channel, self.n_clusters)
                        cluster_label_pre.append(torch.from_numpy(choice_cluster).unsqueeze(0).transpose(1,0))
                        cluster_label = torch.cat(cluster_label_pre, dim = 1)
                        centers_pre.append(torch.from_numpy(initial_centers).unsqueeze(0).transpose(1,0))
                        centers = torch.cat(centers_pre, dim = 1)
                        center_designation_pre.append(cluster_ass.unsqueeze(0).transpose(1,0))
                        center_designation = torch.cat(center_designation_pre, dim = 1)
                    
                    batched_center_designation = list(functions.divide_batches(center_designation, self.batch_size))
                    center_distances, ranks_of_center_distances = functions.rank_channels(centers)
                    end = time.time()
                    hours, minutes, seconds = functions.timer(since, end)
                    print("Time taken {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
                
            print('Epoch : {}/{} Network Loss : {} Clustering Loss : {} Total Loss : {}'.format(epoch+1, 
              self.n_epochs, (total_network_loss/(tr_dataset_length/self.batch_size)), 
          (total_clustering_loss/(tr_dataset_length/self.batch_size)),
          (total_combined_loss/(tr_dataset_length/self.batch_size))))
            
        list_of_centers.append(centers.numpy())
        list_of_ranks_of_center_distances.append(ranks_of_center_distances)
        list_of_center_distances.append(center_distances)
        list_of_losses.append(list_of_network_loss)
        list_of_losses.append(list_of_clustering_loss)
        list_of_losses.append(list_of_total_loss)
        
        return self.network, optimizer, list_of_network_loss, list_of_clustering_loss, list_of_total_loss, list_of_losses, embeddings, labels, list_of_centers, list_of_ranks_of_center_distances, list_of_center_distances
    
    def save_network(self, network):
        torch.save(network.state_dict(), os.path.join(self.path, 'Network.pth'))
        
    def save_loss_list(self, list_of_losses):
        torch.save(list_of_losses, os.path.join(self.path, 'Loss_List.pth'))
        
    def save_optimizer(self, optimizer):
        torch.save(optimizer.state_dict(), os.path.join(self.path, 'Optimizer.pth'))
        
    def save_embeddings(self, embeddings):
        torch.save(embeddings, os.path.join(self.path, 'Embeddings.pth'))
        
    def save_list_of_centers(self, list_of_centers):
        torch.save(list_of_centers, os.path.join(self.path, 'List_of_Centers.pth'))
        
    def save_list_ranks_of_center_distances(self, list_of_ranks_of_center_distances):
        torch.save(list_of_ranks_of_center_distances, os.path.join(self.path, 'List_of_ranks_of_center_distances.pth'))
        
    def training_predictions(self, embeddings):
        km = sklearn.cluster.KMeans(n_clusters = self.n_clusters, init = 'k-means++', n_init = 10, max_iter = 300)
        labels_pred = []
        no_of_channels = embeddings.shape[1]
        for i in range(no_of_channels):
            channel = embeddings[:,i,:].numpy()
            km.fit_predict(channel) 
            cluster_label = km.labels_
            labels_pred.append(cluster_label)
            
        return labels_pred
    
    def calculate_metrics(self, labels_true, labels_pred):
        list_of_nmi = []
        list_of_ari = []
        list_of_acc = []
        list_of_cm = []
        #list_of_accuracy_score = []
        
        for i in range(len(labels_pred)):
            ch_labels_pred = labels_pred[i]
            nmi_score = sklearn.metrics.cluster.normalized_mutual_info_score(labels_true.flatten(), ch_labels_pred)
            list_of_nmi.append(nmi_score)
            
            ari_score = sklearn.metrics.cluster.adjusted_rand_score(labels_true.flatten(), ch_labels_pred)
            list_of_ari.append(ari_score)
            
            cm = sklearn.metrics.confusion_matrix(labels_true.flatten(), ch_labels_pred)
            list_of_cm.append(cm)
            
            acc = functions.metrics.acc(labels_true.flatten(), ch_labels_pred)
            list_of_acc.append(acc)
            
            #accuracy_score = sklearn.metrics.accuracy_score(labels_true, labels_pred)
            #list_of_accuracy_score.append(accuracy_score)
            
        return list_of_nmi, list_of_ari, list_of_acc, list_of_cm
    
    def best_activation_maps(self, list_of_nmi, list_of_ari, list_of_acc, list_of_cm, no_of_map):
        best_nmi = max(list_of_nmi)
        best_ari = max(list_of_ari)
        best_acc = max(list_of_acc)
        
        top_nmi_maps_idx = np.argsort(list_of_nmi)[-no_of_map:]
        top_ari_maps_idx = np.argsort(list_of_ari)[-no_of_map:]
        top_acc_maps_idx = np.argsort(list_of_acc)[-no_of_map:]
        
        return best_nmi, best_ari, best_acc, top_nmi_maps_idx, top_ari_maps_idx, top_acc_maps_idx
    
    def save_train_metrics(self, list_of_nmi, list_of_ari, list_of_acc, list_of_cm, top_nmi, top_ari, top_acc):
        torch.save(list_of_nmi, os.path.join(self.path, 'List_of_nmi.pth'))
        torch.save(list_of_ari, os.path.join(self.path, 'List_of_ari.pth'))
        torch.save(list_of_acc, os.path.join(self.path, 'List_of_acc.pth'))
        torch.save(list_of_cm, os.path.join(self.path, 'List_of_cm.pth'))
        torch.save(top_nmi, os.path.join(self.path, 'Top_NMI.pth'))
        torch.save(top_ari, os.path.join(self.path, 'Top_ARI.pth'))
        torch.save(top_acc, os.path.join(self.path, 'Top_ACC.pth'))
        
    def save_test_metrics(self, list_of_nmi, list_of_ari, list_of_acc, list_of_cm, top_nmi, top_ari, top_acc):
        torch.save(list_of_nmi, os.path.join(self.path, 'List_of_nmi_test.pth'))
        torch.save(list_of_ari, os.path.join(self.path, 'List_of_ari_test.pth'))
        torch.save(list_of_acc, os.path.join(self.path, 'List_of_acc_test.pth'))
        torch.save(list_of_cm, os.path.join(self.path, 'List_of_cm_test.pth'))
        torch.save(top_nmi, os.path.join(self.path, 'Top_NMI_test.pth'))
        torch.save(top_ari, os.path.join(self.path, 'Top_ARI_test.pth'))
        torch.save(top_acc, os.path.join(self.path, 'Top_ACC_test.pth'))
    
    def test_bultmann(self, network, te_loader):
        
        network.eval()
        embedded_representation = []                    
        labels = np.empty((0,1), float)
        
        for batch in te_loader:
            
            #extract the sequence and label from the batch and make predictions and return bottleneck
            
            sequences = batch[:,:,0:self.sequence_length].float()                                     
            batch_labels = batch[:,:,self.sequence_length]
            labels = np.append(labels, batch_labels.numpy(), axis = 0)
            
            predictions, bottleneck = self.network(sequences)
            embedded_representation.append(bottleneck.clone().detach())
            batch_embeddings = torch.cat(embedded_representation)    
                                                          
        #extract embeddings
        embeddings = batch_embeddings
        
        
        return embeddings, labels
    
    def test_teastman(self, network, te_loader):
        
        network.eval()
        embedded_representation = []
        labels = np.empty((1), int)
        
        for data, label in te_loader:
            
            #extract the sequence and label from the batch and make predictions and return bottleneck
        
            sequences = data                                    
            batch_labels = label
            labels = np.append(labels, batch_labels.numpy(), axis = 0)
            target_sequences = sequences.clone()
            predictions, bottleneck = self.network(sequences)
            embedded_representation.append(bottleneck.clone().detach())
            batch_embeddings = torch.cat(embedded_representation)
            
        #extract embeddings
        embeddings = batch_embeddings
        
        return embeddings, labels
    
    def plot_network_loss(self, list_of_losses, color = 'blue'):
        
        network_loss = list_of_losses[0]
        figure = plt.figure(figsize = (8,6))
        plt.plot(network_loss, label = 'Network Loss',color = color)
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Network Loss')
        plt.title('Plot of Network Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.path,'Network Loss.png'), dpi = 300)
        plt.show()
        
    def plot_clustering_loss(self, list_of_losses, color = 'green'):
        
        clustering_loss = list_of_losses[1]
        figure = plt.figure(figsize = (8,6))
        plt.plot(clustering_loss, label = 'Clustering Loss', color = color)
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Clustering Loss')
        plt.title('Plot of Clustering Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'Clustering Loss.png'), dpi = 300)
        plt.show()
        
    def plot_total_loss(self, list_of_losses, color = 'orange'):
        
        total_loss = list_of_losses[2]
        figure = plt.figure(figsize = (8,6))
        plt.plot(total_loss, label = 'Total Loss', color = color)
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.xlabel('Number of Epochs')
        plt.ylabel('Total Loss')
        plt.title('Plot of Total Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'Total Loss.png'), dpi = 300)
        plt.show()
        
    def plot_all_losses(self, list_of_losses, color_one = 'blue', color_two = 'green', color_three = 'orange'):
        
        fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1)
        
        ax1.plot(list_of_losses[0], label = 'Network Loss', color = color_one)
        ax2.plot(list_of_losses[1], label = 'Clustering Loss', color = color_two)
        ax3.plot(list_of_losses[2], label = 'Total Loss', color = color_three)
        
        ax1.legend()
        ax1.grid(b = True, which = 'major', axis = 'both')
        ax1.set_title('Network Loss')
        ax1.set_xlabel('Number of Epochs')
        ax1.set_ylabel('Network Loss')
        
        ax2.legend()
        ax2.grid(b = True, which = 'major', axis = 'both')
        ax2.set_title('Clustering Loss')
        ax2.set_xlabel('Number of Epochs')
        ax2.set_ylabel('Clustering Loss')
        
        ax3.legend()
        ax3.grid(b = True, which = 'major', axis = 'both')
        ax3.set_title('Total Loss')
        ax3.set_xlabel('Number of Epochs')
        ax3.set_ylabel('Total Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'Combined Loss Plot.png'), dpi = 300)
        plt.show()
        
    def plot_two_losses(self, list_of_losses, color_one = 'blue', color_two = 'green'):
        
        fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1)
        
        ax1.plot(list_of_losses[0], label = 'Network Loss', color = color_one)
        ax2.plot(list_of_losses[1], label = 'Clustering Loss', color = color_two)
        
        
        ax1.legend()
        ax1.grid(b = True, which = 'major', axis = 'both')
        ax1.set_title('Network Loss')
        ax1.set_xlabel('Number of Epochs')
        ax1.set_ylabel('Network Loss')
        
        ax2.legend()
        ax2.grid(b = True, which = 'major', axis = 'both')
        ax2.set_title('Clustering Loss')
        ax2.set_xlabel('Number of Epochs')
        ax2.set_ylabel('Clustering Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.path, 'Two Loss Plot.png'), dpi = 300)
        plt.show()
        
    def plot_nmi(self, list_of_nmi, top_nmi_maps_idx, alpha  = 0.8, color = 'indigo'):
        
        top_nmis = []
        
        for i in top_nmi_maps_idx:
            
            top_nmis.append(list_of_nmi[i])
            
        
        figure = plt.figure(figsize = (8,6))
        x_pos = [i for i, _ in enumerate(top_nmi_maps_idx)]
        width = 0.35
        plt.bar(x_pos, top_nmis, align = 'center', alpha = alpha, label = 'NMI', color = color)
        plt.xlabel('Activation Map No')
        plt.ylabel('NMI')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('NMI Scores')
        plt.xticks(x_pos, top_nmi_maps_idx)
        plt.legend()
        plt.savefig(os.path.join(self.path, 'NMI.png'), dpi = 300)
        plt.show()
        
    def plot_ari(self, list_of_ari, top_ari_maps_idx, alpha  = 0.8, color = 'red'):
        
        top_aris = []
        
        for i in top_ari_maps_idx:
            
            top_aris.append(list_of_ari[i])
            
        
        figure = plt.figure(figsize = (8,6))
        x_pos = [i for i, _ in enumerate(top_ari_maps_idx)]
        width = 0.35
        plt.bar(x_pos, top_aris, align = 'center', alpha = alpha, label = 'ARI', color = color)
        plt.xlabel('Activation Map No')
        plt.ylabel('ARI')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('ARI Score')
        plt.xticks(x_pos, top_ari_maps_idx)
        plt.legend()
        plt.savefig(os.path.join(self.path, 'ARI.png'), dpi = 300)
        plt.show()
        
    def plot_acc(self, list_of_acc, top_acc_maps_idx, alpha  = 0.8, color = 'indigo'):
        
        top_accs = []
        
        for i in top_acc_maps_idx:
            
            top_accs.append(list_of_acc[i])
            
        
        figure = plt.figure(figsize = (8,6))
        x_pos = [i for i, _ in enumerate(top_acc_maps_idx)]
        width = 0.35
        plt.bar(x_pos, top_accs, align = 'center', alpha = alpha, label = 'Accuracy', color = color)
        plt.xlabel('Activation Map No')
        plt.ylabel('Accuracy')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('Accuracy')
        plt.xticks(x_pos, top_acc_maps_idx)
        plt.legend()
        plt.savefig(os.path.join(self.path, 'Accuracy.png'), dpi = 300)
        plt.show()
        
    def plot_acc_test(self, list_of_acc, top_acc_maps_idx, alpha  = 0.8, color = 'indigo'):
        
        top_accs = []
        
        for i in top_acc_maps_idx:
            
            top_accs.append(list_of_acc[i])
            
        
        figure = plt.figure(figsize = (8,6))
        x_pos = [i for i, _ in enumerate(top_acc_maps_idx)]
        width = 0.35
        plt.bar(x_pos, top_accs, align = 'center', alpha = alpha, label = 'Accuracy', color = color)
        plt.xlabel('Activation Map No')
        plt.ylabel('Accuracy')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('Testing Accuracy')
        plt.xticks(x_pos, top_acc_maps_idx)
        plt.legend()
        plt.savefig(os.path.join(self.path, 'Accuracy TEST.png'), dpi = 300)
        plt.show()
        
    def plot_top_center_distances(self, list_of_center_distances, no_of_map, alpha = 1.0, color = 'red'):
        
        top_maps_idx = np.argsort(list_of_center_distances[0])[-no_of_map:]
        
        top_maps = []
        
        for i in top_maps_idx:
            
            top_maps.append(list_of_center_distances[0][i])
            
        
        figure = plt.figure(figsize = (8,6))
        x_pos = [i for i, _ in enumerate(top_maps_idx)]
        width = 0.35
        plt.bar(x_pos, top_maps, align = 'center', alpha = alpha, label = 'Center Distances', color = color)
        plt.xlabel('Activation Map No')
        plt.ylabel('Distance between Centers')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('Top Center Distances')
        plt.xticks(x_pos, top_maps_idx)
        plt.legend()
        plt.savefig(os.path.join(self.path, 'Top Center Distances.png'), dpi = 300)
        plt.show()
        
    def plot_center_distances(self, list_of_center_distances, no_of_map, top_acc_maps_idx, alpha = 1.0, color = 'red'):
        
        #top_maps_idx = np.argsort(list_of_center_distances[0])[-no_of_map:]
        
        top_maps = []
        
        for i in top_acc_maps_idx:
            
            top_maps.append(list_of_center_distances[0][i])
            
        
        figure = plt.figure(figsize = (8,6))
        x_pos = [i for i, _ in enumerate(top_acc_maps_idx)]
        width = 0.35
        plt.bar(x_pos, top_maps, align = 'center', alpha = alpha, label = 'Center Distances', color = color)
        plt.xlabel('Activation Map No')
        plt.ylabel('Distance between Centers')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('Distance between Centers')
        plt.xticks(x_pos, top_acc_maps_idx)
        plt.legend()
        plt.savefig(os.path.join(self.path, 'Center Distances.png'), dpi = 300)
        plt.show()
        
    def plot_best_maps_dual_bars(self, list_of_center_distances, list_of_acc, top_acc_maps_idx, alpha = 1.0, color_one = 'red', color_two = 'green'):
        top_accs = []
        top_distances = []
        
        for i in top_acc_maps_idx:
            top_accs.append(list_of_acc[i])
            top_distances.append(list_of_center_distances[0][i])
            
        
        x_pos = np.arange(len(top_accs))
        width = 0.25
        
        
        
        plt.bar(x_pos, top_accs, width, align = 'center', alpha = alpha, label = 'Accuracy', color = color_one)
        plt.bar(x_pos + width, top_distances, width, align = 'center', alpha = alpha, label = 'Center Distance', color = color_two)
        plt.xlabel('Activation Map No')
        plt.ylabel('Accuracy / Distance')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('Accuracy and Center Distance')
        plt.xticks(x_pos + width/2, (top_acc_maps_idx))
        plt.legend()
        plt.savefig(os.path.join(self.path, 'Accuracy and Distance.png'), dpi = 300)
        plt.show()
        
    def plot_nmi_ari(self, list_of_nmi, list_of_ari, top_acc_maps_idx, alpha = 1.0, color_one = 'red', color_two = 'green'):
        top_nmis = []
        top_aris = []
        
        for i in top_acc_maps_idx:
            top_nmis.append(list_of_nmi[i])
            top_aris.append(list_of_ari[i])
            
        
        x_pos = np.arange(len(top_nmis))
        width = 0.25
        
        
        
        plt.bar(x_pos, top_nmis, width, align = 'center', alpha = alpha, label = 'NMI', color = color_one)
        plt.bar(x_pos + width, top_aris, width, align = 'center', alpha = alpha, label = 'ARI', color = color_two)
        plt.xlabel('Activation Map No')
        plt.ylabel('NMI / ARI')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('NMI and ARI')
        plt.xticks(x_pos + width/2, (top_acc_maps_idx))
        plt.legend()
        plt.savefig(os.path.join(self.path, 'NMI and ARI.png'), dpi = 300)
        plt.show()
        
    def plot_nmi_ari_test(self, list_of_nmi, list_of_ari, top_acc_maps_idx, alpha = 1.0, color_one = 'red', color_two = 'green'):
        top_nmis = []
        top_aris = []
        
        for i in top_acc_maps_idx:
            top_nmis.append(list_of_nmi[i])
            top_aris.append(list_of_ari[i])
            
        
        x_pos = np.arange(len(top_nmis))
        width = 0.25
        
        
        
        plt.bar(x_pos, top_nmis, width, align = 'center', alpha = alpha, label = 'NMI', color = color_one)
        plt.bar(x_pos + width, top_aris, width, align = 'center', alpha = alpha, label = 'ARI', color = color_two)
        plt.xlabel('Activation Map No')
        plt.ylabel('NMI / ARI')
        plt.grid(b = True, which = 'major', axis = 'both')
        plt.title('Test NMI and ARI')
        plt.xticks(x_pos + width/2, (top_acc_maps_idx))
        plt.legend()
        plt.savefig(os.path.join(self.path, 'NMI and ARI TEST.png'), dpi = 300)
        plt.show()
        
    def apply_TSNE(self, embeddings, labels_pred, list_of_centers, top_acc_maps_idx, n_components = 2, perplexity = 30.0):
        #top_maps = []
        
        for i in top_acc_maps_idx:
            map_ = embeddings[:, i, :].numpy()
            centers_ = list_of_centers[0][:,i,:]
            tsne = sklearn.manifold.TSNE(n_components = n_components, perplexity = perplexity)
            channel = tsne.fit_transform(map_)
            #centers = tsne.fit_transform(centers_)
            label = labels_pred[i]
            #top_maps.append(tsne_channel)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(channel[label == 0,0], 
                       channel[label == 0, 1], s=1, c = 'green', label = 'Cluster 1')
            ax.scatter(channel[label == 1, 0], 
                       channel[label == 1, 1], s=1, c = 'red', label = 'Cluster 2')
            #ax.scatter(centers[:, 0], centers[:, 1], c='black', s=10, alpha=0.5)
                
            plt.grid(b = True, which = 'major', axis = 'both')
            plt.title('Activation Map {}'.format(i))
            plt.legend()
            plt.savefig(os.path.join(self.path, 'TSNE Plot Channel {}.png'.format(i)), dpi = 300)
            plt.show()
            
        
    
    def plot_bottleneck(self, top_maps, top_acc_maps_idx, color_one = 'green', color_two = 'red'):
        for i in range(len(top_maps)):
            channel = top_maps[i]
            label = self.labels_pred[i]
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(channel[label == 0,0], 
                       channel[label == 0, 1], s=1, c = 'green', label = 'Cluster 1')
            ax.scatter(channel[label == 1, 0], 
                       channel[label == 1, 1], s=1, c = 'red', label = 'Cluster 2')
                
            plt.grid(b = True, which = 'major', axis = 'both')
            plt.title('Activation Map {}'.format(i+1))
            plt.legend()
            plt.savefig('TSNE Plot Channel {}.png'.format(i+1), dpi = 300)
            plt.show()  
