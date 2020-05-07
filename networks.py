import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

#3 layer encoder convolutional autoencoder CONTROL
class CAE(nn.Module):
    def __init__(self):
        super(CAE, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 2)
        self.conv2 = nn.Conv1d(in_channels = 4, out_channels = 32, kernel_size = 5, stride = 2)
        self.conv3 = nn.Conv1d(in_channels = 8, out_channels = 12, kernel_size = 3, stride = 1)
        self.deconv3 = nn.ConvTranspose1d(in_channels = 12, out_channels = 8, kernel_size = 3, stride = 1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = 8, out_channels = 4, kernel_size = 5, stride = 2)
        self.deconv1 = nn.ConvTranspose1d(in_channels = 4, out_channels = 1, kernel_size = 5, stride = 2, output_padding = 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        extra_out = x
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        #x = torch.sigmoid(self.deconv1(x))
        
        return x, extra_out
    
    

#4 layer encoder convolutional autoencoder CONTROL
class CAE1(nn.Module):
    def __init__(self):
        super(CAE1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 4, kernel_size = 5, stride = 1)
        self.conv2 = nn.Conv1d(in_channels = 4, out_channels = 8, kernel_size = 5, stride = 1)
        self.conv3 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 5, stride = 1)
        self.conv4 = nn.Conv1d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1)
        self.deconv4 = nn.ConvTranspose1d(in_channels = 32, out_channels = 16, kernel_size = 5, stride = 1)
        self.deconv3 = nn.ConvTranspose1d(in_channels = 16, out_channels = 8, kernel_size = 5, stride = 1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = 8, out_channels = 4, kernel_size = 5, stride = 1)
        self.deconv1 = nn.ConvTranspose1d(in_channels = 4, out_channels = 1, kernel_size = 5, stride = 1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        extra_out = x
        x = F.relu(self.deconv4(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = F.sigmoid(self.deconv1(x))
        
        return x, extra_out
    
    
#3 layer convolutional autoencoder with linear bottleneck CONTROL
class CAE2(nn.Module):
    def __init__(self):
        super(CAE2, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 1, out_channels = 4, kernel_size = 5, stride = 2)
        self.conv2 = nn.Conv1d(in_channels = 4, out_channels = 8, kernel_size = 5, stride = 2)
        self.conv3 = nn.Conv1d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 1)
        self.embedding1 = nn.Linear(in_features = 16*3, out_features = 16)
        self.embedding = nn.Linear(in_features = 16, out_features = 3)
        self.deembedding = nn.Linear(in_features = 3, out_features = 16)
        self.deembedding1 = nn.Linear(in_features = 16, out_features = 16*3)
        self.deconv3 = nn.ConvTranspose1d(in_channels = 16, out_channels = 8, kernel_size = 3, stride = 1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = 8, out_channels = 4, kernel_size = 5, stride = 2)
        self.deconv1 = nn.ConvTranspose1d(in_channels = 4, out_channels = 1, kernel_size = 5, stride = 2, output_padding = 1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.embedding1(x))
        x = F.relu(self.embedding(x))
        extra_out = x
        x = F.relu(self.deembedding(x))
        x = F.relu(self.deembedding1(x))
        x = x.view(x.size(0), 16, 3)
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv2(x))
        x = F.sigmoid(self.deconv1(x))
        
        return x, extra_out
    
#3 layer CAE

class CAE_3(nn.Module):
    
    def __init__(self, input_channels, filter_multiplier, kernel_size, leaky = True, neg_slope = 0.01, activations = True):
        super(CAE_3, self).__init__()
        self.input_channels = input_channels
        self.filter_multiplier = filter_multiplier
        self.kernel_size = kernel_size
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope = neg_slope)
        else:
            self.relu = nn.ReLU(inplace = False)
        self.activations = activations
        self.conv1 = nn.Conv1d(in_channels = input_channels, out_channels = input_channels*filter_multiplier,
                               kernel_size = kernel_size, stride = 2)
        self.conv2 = nn.Conv1d(in_channels = input_channels*filter_multiplier, 
                               out_channels = input_channels*filter_multiplier*2,
                               kernel_size = kernel_size, stride = 2)
        self.conv3 = nn.Conv1d(in_channels = input_channels*filter_multiplier*2, 
                               out_channels = input_channels*filter_multiplier*4,
                               kernel_size = kernel_size, stride = 1)
        self.deconv3 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier*4, 
                                          out_channels = input_channels*filter_multiplier*2,
                                          kernel_size = kernel_size, stride = 1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier*2, 
                                          out_channels = input_channels*filter_multiplier,
                                          kernel_size = kernel_size, stride = 2, output_padding = 1)
        self.deconv1 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier, 
                                          out_channels = input_channels, kernel_size = kernel_size, 
                                          stride = 2, output_padding = 1)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        extra_out = x
        x = self.deconv3(x)
        x = self.relu3_2(x)
        x = self.deconv2(x)
        x = self.relu2_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.sig(x)
        
        return x, extra_out 
    
#3 layer CAE batch normalized version

class CAE_3_bn(nn.Module):
    
    def __init__(self, input_channels, filter_multiplier, kernel_size, leaky = True, neg_slope = 0.01, activations = True):
        super(CAE_3_bn, self).__init__()
        self.input_channels = input_channels
        self.filter_multiplier = filter_multiplier
        self.kernel_size = kernel_size
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope = neg_slope)
        else:
            self.relu = nn.ReLU(inplace = False)
        self.activations = activations
        self.conv1 = nn.Conv1d(in_channels = input_channels, out_channels = filter_multiplier,
                               kernel_size = kernel_size, stride = 2)
        self.bn1_1 = nn.BatchNorm1d(num_features = filter_multiplier, eps = 1e-5,
                                    momentum = 0.1)
        self.conv2 = nn.Conv1d(in_channels = filter_multiplier, 
                               out_channels = filter_multiplier*2,
                               kernel_size = kernel_size, stride = 2)
        self.bn2_1 = nn.BatchNorm1d(num_features = filter_multiplier*2, eps = 1e-5,
                                    momentum = 0.1)
        self.conv3 = nn.Conv1d(in_channels = filter_multiplier*2, 
                               out_channels = filter_multiplier*4,
                               kernel_size = kernel_size, stride = 1)
        self.bn3_1 = nn.BatchNorm1d(num_features = filter_multiplier*4, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv3 = nn.ConvTranspose1d(in_channels = filter_multiplier*4, 
                                          out_channels = filter_multiplier*2,
                                          kernel_size = kernel_size, stride = 1)
        self.bn3_2 = nn.BatchNorm1d(num_features = filter_multiplier*2, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = filter_multiplier*2, 
                                          out_channels = filter_multiplier,
                                          kernel_size = kernel_size, stride = 2, output_padding = 1)
        self.bn2_2 = nn.BatchNorm1d(num_features = filter_multiplier, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv1 = nn.ConvTranspose1d(in_channels = filter_multiplier, 
                                          out_channels = input_channels, kernel_size = kernel_size, 
                                          stride = 2, output_padding = 1)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        extra_out = x
        x = self.deconv3(x)
        x = self.relu3_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu2_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.sig(x)
        
        return x, extra_out
    
#4 layer CAE
    
class CAE_4(nn.Module):
    
    def __init__(self, input_channels, filter_multiplier, leaky = True, neg_slope = 0.01, activations = True):
        super(CAE_4, self).__init__()
        self.input_channels = input_channels
        self.filter_multiplier = filter_multiplier
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope = neg_slope)
        else:
            self.relu = nn.ReLU(inplace = False)
        self.activations = activations
        self.conv1 = nn.Conv1d(in_channels = input_channels, out_channels = input_channels*filter_multiplier,
                               kernel_size = 5, stride = 2)
        self.conv2 = nn.Conv1d(in_channels = input_channels*filter_multiplier, 
                               out_channels = input_channels*filter_multiplier*2,
                               kernel_size = 5, stride = 2)
        self.conv3 = nn.Conv1d(in_channels = input_channels*filter_multiplier*2, 
                               out_channels = input_channels*filter_multiplier*4,
                               kernel_size = 4, stride = 1)
        self.conv4 = nn.Conv1d(in_channels = input_channels*filter_multiplier*4, 
                               out_channels = input_channels*filter_multiplier*8,
                               kernel_size = 4, stride = 1)
        self.deconv4 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier*8, 
                                          out_channels = input_channels*filter_multiplier*4,
                                          kernel_size = 4, stride = 1)
        self.deconv3 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier*4, 
                                          out_channels = input_channels*filter_multiplier*2,
                                          kernel_size = 4, stride = 1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier*2, 
                                          out_channels = input_channels*filter_multiplier,
                                          kernel_size = 5, stride = 2, output_padding = 0)
        self.deconv1 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier, 
                                          out_channels = input_channels, kernel_size = 5, 
                                          stride = 2, output_padding = 1)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        extra_out = x
        x = self.deconv4(x)
        x = self.relu4_2(x)
        x = self.deconv3(x)
        x = self.relu3_2(x)
        x = self.deconv2(x)
        x = self.relu2_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.sig(x)
        
        return x, extra_out
    
#4 layer CAE batch normalized version
    
class CAE_4_bn(nn.Module):
    
    def __init__(self, input_channels, filter_multiplier, kernel_size, leaky = True, neg_slope = 0.01, activations = True):
        super(CAE_4_bn, self).__init__()
        self.input_channels = input_channels
        self.filter_multiplier = filter_multiplier
        self.kernel_size = kernel_size
        if leaky:
            self.relu = nn.GELU()
        else:
            self.relu = nn.ReLU(inplace = False)
        self.activations = activations
        self.conv1 = nn.Conv1d(in_channels = input_channels, out_channels = filter_multiplier,
                               kernel_size = kernel_size, stride = 2)
        self.bn1_1 = nn.BatchNorm1d(num_features = filter_multiplier, eps = 1e-5,
                                    momentum = 0.1)
        self.conv2 = nn.Conv1d(in_channels = filter_multiplier, 
                               out_channels = filter_multiplier*2,
                               kernel_size = kernel_size, stride = 2)
        self.bn2_1 = nn.BatchNorm1d(num_features = filter_multiplier*2, eps = 1e-5,
                                    momentum = 0.1)
        self.conv3 = nn.Conv1d(in_channels = filter_multiplier*2, 
                               out_channels = filter_multiplier*4,
                               kernel_size = kernel_size, stride = 1)
        self.bn3_1 = nn.BatchNorm1d(num_features = filter_multiplier*4, eps = 1e-5,
                                    momentum = 0.1)
        self.conv4 = nn.Conv1d(in_channels = filter_multiplier*4, 
                               out_channels = filter_multiplier*8,
                               kernel_size = kernel_size, stride = 1)
        self.bn4_1 = nn.BatchNorm1d(num_features = filter_multiplier*8, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv4 = nn.ConvTranspose1d(in_channels = filter_multiplier*8, 
                                          out_channels = filter_multiplier*4,
                                          kernel_size = kernel_size, stride = 1)
        self.bn4_2 = nn.BatchNorm1d(num_features = filter_multiplier*4, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv3 = nn.ConvTranspose1d(in_channels = filter_multiplier*4, 
                                          out_channels = filter_multiplier*2,
                                          kernel_size = kernel_size, stride = 1)
        self.bn3_2 = nn.BatchNorm1d(num_features = filter_multiplier*2, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = filter_multiplier*2, 
                                          out_channels = filter_multiplier,
                                          kernel_size = kernel_size, stride = 2, output_padding = 0)
        self.bn2_2 = nn.BatchNorm1d(num_features = filter_multiplier, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv1 = nn.ConvTranspose1d(in_channels = filter_multiplier, 
                                          out_channels = input_channels, kernel_size = kernel_size, 
                                          stride = 2, output_padding = 1)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.bn4_1(x)
        extra_out = x
        x = self.deconv4(x)
        x = self.relu4_2(x)
        x = self.bn4_2(x)
        x = self.deconv3(x)
        x = self.relu3_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu2_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.sig(x)
        
        return x, extra_out
    
#5 layer CAE
    
class CAE_5(nn.Module):
    
    def __init__(self, input_channels, filter_multiplier, leaky = True, neg_slope = 0.01, activations = True):
        super(CAE_5, self).__init__()
        self.input_channels = input_channels
        self.filter_multiplier = filter_multiplier
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope = neg_slope)
        else:
            self.relu = nn.ReLU(inplace = False)
        self.activations = activations
        self.conv1 = nn.Conv1d(in_channels = input_channels, out_channels = input_channels*filter_multiplier,
                               kernel_size = 5, stride = 2)
        self.conv2 = nn.Conv1d(in_channels = input_channels*filter_multiplier, 
                               out_channels = input_channels*filter_multiplier*2,
                               kernel_size = 5, stride = 2)
        self.conv3 = nn.Conv1d(in_channels = input_channels*filter_multiplier*2, 
                               out_channels = input_channels*filter_multiplier*4,
                               kernel_size = 5, stride = 1)
        self.conv4 = nn.Conv1d(in_channels = input_channels*filter_multiplier*4, 
                               out_channels = input_channels*filter_multiplier*8,
                               kernel_size = 4, stride = 1)
        self.conv5 = nn.Conv1d(in_channels = input_channels*filter_multiplier*8, 
                               out_channels = input_channels*filter_multiplier*12,
                               kernel_size = 4, stride = 1)
        self.deconv5 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier*12, 
                                          out_channels = input_channels*filter_multiplier*8,
                                          kernel_size = 4, stride = 1)
        self.deconv4 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier*8, 
                                          out_channels = input_channels*filter_multiplier*4,
                                          kernel_size = 4, stride = 1)
        self.deconv3 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier*4, 
                                          out_channels = input_channels*filter_multiplier*2,
                                          kernel_size = 5, stride = 1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier*2, 
                                          out_channels = input_channels*filter_multiplier,
                                          kernel_size = 5, stride = 2, output_padding = 1)
        self.deconv1 = nn.ConvTranspose1d(in_channels = input_channels*filter_multiplier, 
                                          out_channels = input_channels, kernel_size = 5, 
                                          stride = 2, output_padding = 1)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu5_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.relu5_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.conv5(x)
        x = self.relu5_1(x)
        extra_out = x
        x = self.deconv5(x)
        x = self.relu5_2(x)
        x = self.deconv4(x)
        x = self.relu4_2(x)
        x = self.deconv3(x)
        x = self.relu3_2(x)
        x = self.deconv2(x)
        x = self.relu2_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.sig(x)
        
        return x, extra_out 
    
#5 layer CAE batch normalized version
    
class CAE_5_bn(nn.Module):
    
    def __init__(self, input_channels, filter_multiplier, leaky = True, neg_slope = 0.01, activations = True):
        super(CAE_5_bn, self).__init__()
        self.input_channels = input_channels
        self.filter_multiplier = filter_multiplier
        if leaky:
            self.relu = nn.LeakyReLU(negative_slope = neg_slope)
        else:
            self.relu = nn.ReLU(inplace = False)
        self.activations = activations
        self.conv1 = nn.Conv1d(in_channels = input_channels, out_channels = filter_multiplier,
                               kernel_size = 5 , stride = 2)
        self.bn1_1 = nn.BatchNorm1d(num_features = filter_multiplier, eps = 1e-5,
                                    momentum = 0.1)
        self.conv2 = nn.Conv1d(in_channels = filter_multiplier, 
                               out_channels = filter_multiplier*2,
                               kernel_size = 5, stride = 2)
        self.bn2_1 = nn.BatchNorm1d(num_features = filter_multiplier*2, eps = 1e-5,
                                    momentum = 0.1)
        self.conv3 = nn.Conv1d(in_channels = filter_multiplier*2, 
                               out_channels = filter_multiplier*4,
                               kernel_size = 5, stride = 1)
        self.bn3_1 = nn.BatchNorm1d(num_features = filter_multiplier*4, eps = 1e-5,
                                    momentum = 0.1)
        self.conv4 = nn.Conv1d(in_channels = filter_multiplier*4, 
                               out_channels = filter_multiplier*8,
                               kernel_size = 4, stride = 1)
        self.bn4_1 = nn.BatchNorm1d(num_features = filter_multiplier*8, eps = 1e-5,
                                    momentum = 0.1)
        self.conv5 = nn.Conv1d(in_channels = filter_multiplier*8, 
                               out_channels = filter_multiplier*12,
                               kernel_size = 4, stride = 1)
        self.bn5_1 = nn.BatchNorm1d(num_features = filter_multiplier*12, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv5 = nn.ConvTranspose1d(in_channels = filter_multiplier*12, 
                                          out_channels = input_channels*filter_multiplier*8,
                                          kernel_size = 4, stride = 1)
        self.bn5_2 = nn.BatchNorm1d(num_features = filter_multiplier*8, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv4 = nn.ConvTranspose1d(in_channels = filter_multiplier*8, 
                                          out_channels = filter_multiplier*4,
                                          kernel_size = 4, stride = 1)
        self.bn4_2 = nn.BatchNorm1d(num_features = filter_multiplier*4, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv3 = nn.ConvTranspose1d(in_channels = filter_multiplier*4, 
                                          out_channels = filter_multiplier*2,
                                          kernel_size = 5, stride = 1)
        self.bn3_2 = nn.BatchNorm1d(num_features = filter_multiplier*2, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv2 = nn.ConvTranspose1d(in_channels = filter_multiplier*2, 
                                          out_channels = filter_multiplier,
                                          kernel_size = 5, stride = 2, output_padding = 1)
        self.bn2_2 = nn.BatchNorm1d(num_features = filter_multiplier, eps = 1e-5,
                                    momentum = 0.1)
        self.deconv1 = nn.ConvTranspose1d(in_channels = filter_multiplier, 
                                          out_channels = input_channels, kernel_size = 5, 
                                          stride = 2, output_padding = 1)
        self.relu1_1 = copy.deepcopy(self.relu)
        self.relu2_1 = copy.deepcopy(self.relu)
        self.relu3_1 = copy.deepcopy(self.relu)
        self.relu4_1 = copy.deepcopy(self.relu)
        self.relu5_1 = copy.deepcopy(self.relu)
        self.relu1_2 = copy.deepcopy(self.relu)
        self.relu2_2 = copy.deepcopy(self.relu)
        self.relu3_2 = copy.deepcopy(self.relu)
        self.relu4_2 = copy.deepcopy(self.relu)
        self.relu5_2 = copy.deepcopy(self.relu)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1_1(x)
        x = self.bn1_1(x)
        x = self.conv2(x)
        x = self.relu2_1(x)
        x = self.bn2_1(x)
        x = self.conv3(x)
        x = self.relu3_1(x)
        x = self.bn3_1(x)
        x = self.conv4(x)
        x = self.relu4_1(x)
        x = self.bn4_1(x)
        x = self.conv5(x)
        x = self.relu5_1(x)
        x = self.bn5_1(x)
        extra_out = x
        x = self.deconv5(x)
        x = self.relu5_2(x)
        x = self.bn5_2(x)
        x = self.deconv4(x)
        x = self.relu4_2(x)
        x = self.bn4_2(x)
        x = self.deconv3(x)
        x = self.relu3_2(x)
        x = self.bn3_2(x)
        x = self.deconv2(x)
        x = self.relu2_2(x)
        x = self.bn2_2(x)
        x = self.deconv1(x)
        if self.activations:
            x = self.sig(x)
        
        return x, extra_out
    
   
    
  
    
    
    
    
    
    
    
    
    