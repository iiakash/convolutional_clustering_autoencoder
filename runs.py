"""Datasets used: Bultmann (labeled) and Tenneessee Eastman"""

import time
import numpy as np
        
import anomaly_detection
import networks
import torch
import torch.nn as nn
import helpers

#take inputs

mode = input('Run Mode: ')
network_criterion = nn.MSELoss()
clustering_criterion = nn.MSELoss()
input_channels = int(input('No of Input Channels: '))
kernel_size = int(input('Kernel Size: '))
filter_multiplier = int(input('No of output channels in first layer: '))
n_clusters = int(input('No of Clusters: '))
cluster_update_interval = int(input('Cluster Center Update Interval: '))
no_of_clustering_channels = int(input('No of Clustering Channels: '))
n_epochs = int(input('No of Epochs: '))
no_of_pretrain_epochs = int(input('No of AE Pretrain Epochs: '))
batch_size = int(input('Batch Size: '))
lr = float(input('Learning Rate: '))
alpha = float(input('Alpha: '))
downsampling_step = int(input('Downsampling Step: '))
sequence_length = int(input('Time Series Sequence Length: '))
no_of_map = int(input('No of Best Activation Maps for Plotting: '))

#start timer



#initiate the suitable network

network = networks.CAE_3_bn(input_channels = input_channels, filter_multiplier = filter_multiplier, kernel_size = kernel_size, leaky = True, neg_slope = 0.01, activations = False)


since = time.time()

if mode == 'Bultmann':
    #initiate the run class
    run = anomaly_detection.AnomalyDetection(mode, network, network_criterion, n_clusters, 
               clustering_criterion, cluster_update_interval,
               no_of_clustering_channels, n_epochs, no_of_pretrain_epochs, 
               batch_size, lr, alpha,downsampling_step, sequence_length, kernel_size)
    
    #load the data
    
    train_loader, train_data_len, test_loader, test_data_len = run.load_bultmann_data(normalize = True)
      
    #train the network

    network, optimizer, list_of_network_loss, list_of_clustering_loss,list_of_total_loss, list_of_losses, embeddings,labels, list_of_centers, list_of_ranks_of_center_distances, list_of_center_distances = run.train_bultmann(train_loader, train_data_len, Adam = False, scheduler = False)
    
    #training complete
    
    #final kmeans on the trained embedded space
        
    labels_pred = run.training_predictions(embeddings)
        
    #compute the training metrics
        
    list_of_nmi, list_of_ari, list_of_acc, list_of_cm = run.calculate_metrics(labels, labels_pred)
    
    #best metrics and top embedding channels
        
    best_training_nmi, best_training_ari, best_training_acc, top_nmi_maps_idx, top_ari_maps_idx, top_acc_maps_idx = run.best_activation_maps(list_of_nmi, list_of_ari, list_of_acc, list_of_cm, no_of_map)
    
    test_embeddings, test_labels_true = run.test_teastman(network, test_loader)
    
    test_labels_pred = run.training_predictions(embeddings)
        
    #compute the training metrics
        
    list_of_nmi_test, list_of_ari_test, list_of_acc_test, list_of_cm_test = run.calculate_metrics(test_labels_true, test_labels_pred)
    
    #best metrics and top embedding channels
        
    best_testing_nmi, best_testing_ari, best_testing_acc, top_nmi_maps_idx_test, top_ari_maps_idx_test, top_acc_maps_idx_test = run.best_activation_maps(list_of_nmi_test, list_of_ari_test, list_of_acc_test, list_of_cm_test, no_of_map)
    
elif mode == 'TEastman':
    #initiate the run class
    run = anomaly_detection.AnomalyDetection(mode, network, network_criterion, n_clusters, 
               clustering_criterion, cluster_update_interval,
               no_of_clustering_channels, n_epochs, no_of_pretrain_epochs, 
               batch_size, lr, alpha,downsampling_step, sequence_length, kernel_size)
    
    #load the data 
    
    train_loader, test_loader, _ = run.load_teastman_data(normalize = True)
    
    #train the network
    
    network, optimizer, list_of_network_loss, list_of_clustering_loss,list_of_total_loss, list_of_losses, embeddings, labels, list_of_centers, list_of_ranks_of_center_distances, list_of_center_distances = run.train_teastman(train_loader, 9920, Adam = False, scheduler = True)
    
    #training complete
    
    #final kmeans on the trained embedded space
        
    labels_pred = run.training_predictions(embeddings)
    
    #compute the training metrics
    labels[np.where(labels > 0)] = 1  
    list_of_nmi, list_of_ari, list_of_acc, list_of_cm = run.calculate_metrics(labels[1:], labels_pred)
    
    #best metrics and top embedding channels
        
    best_training_nmi, best_training_ari, best_training_acc, top_nmi_maps_idx, top_ari_maps_idx, top_acc_maps_idx = run.best_activation_maps(list_of_nmi, list_of_ari, list_of_acc, list_of_cm, no_of_map)

    test_embeddings, test_labels_true = run.test_teastman(network, test_loader)
    
    test_labels_pred = run.training_predictions(test_embeddings)
        
    #compute the training metrics
    test_labels_true[np.where(test_labels_true > 0)] = 1   
    list_of_nmi_test, list_of_ari_test, list_of_acc_test, list_of_cm_test = run.calculate_metrics(test_labels_true[1:], test_labels_pred)
    
    #best metrics and top embedding channels
        
    best_testing_nmi, best_testing_ari, best_testing_acc, top_nmi_maps_idx_test, top_ari_maps_idx_test, top_acc_maps_idx_test = run.best_activation_maps(list_of_nmi_test, list_of_ari_test, list_of_acc_test, list_of_cm_test, no_of_map)
    
elif mode == 'TEastmanFull':
    
    #initiate the run class
    run = anomaly_detection.AnomalyDetection(mode, network, network_criterion, n_clusters, 
               clustering_criterion, cluster_update_interval,
               no_of_clustering_channels, n_epochs, no_of_pretrain_epochs, 
               batch_size, lr, alpha,downsampling_step, sequence_length, kernel_size)
    
    #load the data
    
    data_set, train_loader = run.load_teastman_full_data(train = True, normalize = True)
    
    #train the network
    
    network, optimizer, list_of_network_loss, list_of_clustering_loss,list_of_total_loss, list_of_losses, embeddings, labels, list_of_centers,list_of_ranks_of_center_distances, list_of_center_distances = run.train_teastman(train_loader, 9260, Adam = False, scheduler = True)


    
#save 
run.save_network(network)
run.save_optimizer(optimizer)
run.save_embeddings(embeddings)
run.save_embeddings(test_embeddings)
run.save_loss_list(list_of_losses)
run.save_list_of_centers(list_of_centers)
run.save_list_ranks_of_center_distances(list_of_ranks_of_center_distances)
run.save_train_metrics(list_of_nmi, list_of_ari, list_of_acc, list_of_cm, top_nmi_maps_idx, top_ari_maps_idx, top_acc_maps_idx)
run.save_test_metrics(list_of_nmi_test, list_of_ari_test, list_of_acc_test, list_of_cm_test, top_nmi_maps_idx_test, top_ari_maps_idx_test, top_acc_maps_idx_test)
       
#plot training results
run.plot_network_loss(list_of_losses, color = 'blue')
run.plot_clustering_loss(list_of_losses, color = 'green')
run.plot_total_loss(list_of_losses, color = 'brown')
run.plot_two_losses(list_of_losses, color_one = 'blue', color_two = 'green')
run.plot_all_losses(list_of_losses, color_one = 'blue', color_two = 'green', color_three = 'brown')

run.plot_nmi(list_of_nmi, top_nmi_maps_idx, alpha = 1.0, color = 'green')
run.plot_ari(list_of_ari, top_ari_maps_idx, alpha = 1.0, color = 'blue')
run.plot_acc(list_of_acc, top_acc_maps_idx, alpha = 1.0, color = 'red')

run.plot_top_center_distances(list_of_center_distances, no_of_map, alpha=0.7,color = 'blue')
run.plot_center_distances(list_of_center_distances, no_of_map, top_acc_maps_idx, alpha = 1.0, color = 'red')
run.plot_best_maps_dual_bars(list_of_center_distances, list_of_acc, top_acc_maps_idx, alpha = 1.0, color_one = 'red', color_two = 'green')
run.plot_nmi_ari(list_of_nmi, list_of_ari, top_acc_maps_idx)
#plot testing results
run.plot_acc_test(list_of_acc_test, top_acc_maps_idx_test, alpha = 0.8, color = 'red')
run.plot_nmi_ari_test(list_of_nmi_test, list_of_ari_test,top_acc_maps_idx_test, alpha = 0.8)

end = time.time()

hours, minutes, seconds = helpers.timer(since, end)
print("Time taken {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
tsne_since = time.time()
run.apply_TSNE(embeddings, labels_pred, list_of_centers, top_acc_maps_idx, n_components = 2, perplexity = 30.0)
tsne_end = time.time()
hours, minutes, seconds = helpers.timer(tsne_since, tsne_end)
print("Time taken {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    

    
    