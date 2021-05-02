import torch
import torch.nn as nn
import torchvision
from collections import OrderedDict

class SeaNormaBlock(nn.Module):
    def __init__(self, input_size, num_classes,dropout=0.5):
        super(SeaNormaBlock, self).__init__()
        self.fc = nn.Linear(input_size, input_size)
        self.bn = nn.BatchNorm1d(input_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x
    
    
class ResNet(nn.Module):
    resnet_layer_dims = [64,64,64,64,256,512,1024,2048,2048,1000]
    
    def __init__(self, num_classes=200, hidden_size=2048, num_blocks=1, requires_grad=False, layer_cutoff=8):
        '''
        1. Creates Resnet101 instance cutoff at layer_cutoff (exclusive)
        2. Adds a (1,1) average pool layer at end (like in original ResNet, probably other ways we can do this)
        3. Adds num_block number of blocks to the end of ResNet
            One block:
                Fully connected layer (out_dim = hidden_size)
                Batchnorm
                Relu
                Dropout
        4. Adds Fully connected layer with (out_dim = num_classes)
        
        Resnet101 Details:
        10 layers:
        Layer 1: 64 channel output conv
        Layer 2: Batchnorm Layer
        Layer 3: Relu
        Layer 4: 2d Maxpool 
        Layer 5: 256 channel output with 2 bottleneck blocks
        Layer 6: 512 channel output with 3 bottleneck blocks
        Layer 7: 1024 channel output with 22 bottleneck blocks
        Layer 8: 2048 channel output with 2 bottleneck blocks
        Layer 9: (1x1) average pool.
        Layer 10: (2048 in, 1000 out) FC layer
        '''
        super(ResNet, self).__init__()
        
        pretrained_resnet = torchvision.models.resnet101(pretrained=True)
        
        for p in pretrained_resnet.parameters():
            p.requires_grad = requires_grad
            
        modules = list(pretrained_resnet.children())[0:layer_cutoff]
        
        self.model = nn.Sequential(*modules)
        self.model.add_module("Average Pool", nn.AdaptiveAvgPool2d((1,1)))
        
        #map output layer to output feature size
        #i.e. layer 5 cutoff corresponds to 256 feature size
        in_dim = ResNet.resnet_layer_dims[layer_cutoff-1]
        
        for i in range(num_blocks):
            self.model.add_module("Block" + str(i), SeaNormaBlock(in_dim, hidden_size))
            in_dim = hidden_size
            
        self.model.add_module("FC", nn.Linear(in_dim, num_classes))
    
    def forward(self, x):
        x = self.model(x)
        return x
   
        
class Inception(nn.Module):
    
    inception_layer_dims = [32, 32, 64, 64, 80,192, 192, 256, 288, 288, 768, 768, 768, 768, 768, None, 1280, 2048, 2048, 2048, 2048, 1000]
    
    inception_layer_dims_no_aux = [32, 32, 64, 64, 80,192, 192, 256, 288, 288, 768, 768, 768, 768, 768, 1280, 2048, 2048, 2048, 2048, 1000]
    
    def __init__(self, num_classes=200, hidden_size=2048, num_blocks=1, requires_grad=False, layer_cutoff=20, use_aux=False):
        '''
        1. Creates InceptionV3 instance cutoff at layer_cutoff
        2. Adds a (1,1) average pool layer at end (like in original ResNet, probably other ways we can do this)
        3. Adds num_block number of blocks to the end of ResNet
            One block:
                Fully connected layer (out_dim = hidden_size)
                Batchnorm
                Relu
                Dropout
        4. Adds Fully connected layer with (out_dim = num_classes)
        
        InceptionV3 Details:
        22 total layers:
        layer     1: 1 conv layer (out channels = 32)
        layer   2-4: 2 conv layers -> max pool layer (out channels = [32,64,64])
        layer   5-7: 2 conv layers -> max pool layer (out channels = [80,192,192])
        layer  8-10: InceptionA blocks (out channels = [256, 288, 288])
        layer    11: InceptionB block (out channels = 768)
        layer 12-15: InceptionC blocks (out channels = [768, 768, 768, 768])
        layer    16: Tangential InceptionAux block - can look more deeply into this to extract features
        layer    17: InceptionD block (out channels = 1280)
        layer 18-19: InceptionE block (out channels = [2048, 2048])
        layer    20: Average pool layer
        layer    21: Dropout layer
        layer    22: Fully connected layer (in_dim=2048, out_dim=1000)
        '''
        
        super(Inception, self).__init__()
        
        pretrained_inception = torchvision.models.inception_v3(pretrained=True, aux_logits=use_aux)
        
        for p in pretrained_inception.parameters():
            p.requires_grad = requires_grad
            
        modules = list(pretrained_inception.children())[0:layer_cutoff]
        
        self.model = nn.Sequential(*modules)
        self.model.add_module("Average Pool", nn.AdaptiveAvgPool2d((1,1)))
        
        #map output layer to output feature size
        #i.e. layer 5 cutoff corresponds to 256 feature size
        
        if use_aux:
            in_dim = Inception.inception_layer_dims[layer_cutoff-1]
        else:
            in_dim = Inception.inception_layer_dims_no_aux[layer_cutoff-1]
        
        for i in range(num_blocks):
            self.model.add_module("Block" + str(i), SeaNormaBlock(in_dim, hidden_size))
            in_dim = hidden_size
            
        self.model.add_module("FC", nn.Linear(in_dim, num_classes))

    def forward(self, x):
        x = self.model.forward(x)
        return x
    
    
    