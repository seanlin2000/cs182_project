import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.functional as F
from collections import OrderedDict


class SeaNormaBlock(nn.Module):
    def __init__(self, input_size, out_size, dropout=0.5):
        super().__init__()
        self.fc = nn.Linear(input_size, out_size)
        self.bn = nn.BatchNorm1d(out_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class CaptainAmerica(nn.Module):
    vgg = []

    def __init__(self, num_classes=200, hidden_size=2048, num_blocks=1, requires_grad=False, layer_cutoff=-1,
                 extract_features=False, pre_trained=True):
        """
        1. Creates a VGG transfer instance
        """
        super().__init__()

        pretrained_vgg = torchvision.models.vgg19_bn(pretrained=pre_trained)

        for p in pretrained_vgg.parameters():
            p.requires_grad = False

        feature_layers = list(pretrained_vgg.features.children())
        pool_layer = [pretrained_vgg.avgpool]
        flatten_layer = [nn.Flatten()]
        classifier_layers = list(pretrained_vgg.classifier.children())
        modules = feature_layers + pool_layer + flatten_layer + classifier_layers

        modules = modules[:layer_cutoff]

        self.model = nn.Sequential(*modules)

        in_dim = 4096
        self.out_dim = in_dim

        if not extract_features:
            for i in range(num_blocks):
                self.model.add_module("Block" + str(i), SeaNormaBlock(in_dim, hidden_size))
                in_dim = hidden_size

            self.model.add_module("Logits", nn.Linear(in_dim, num_classes))
            self.out_dim = num_classes

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def get_out_dim(self):
        return self.out_dim


class Thor(nn.Module):
    resnet_layer_dims = [64, 64, 64, 64, 256, 512, 1024, 2048, 2048, 1000]

    def __init__(self, num_classes=200, hidden_size=2048, num_blocks=1, requires_grad=False, layer_cutoff=8,
                 extract_features=False, pre_trained=True):
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
        super().__init__()

        pretrained_resnet = torchvision.models.resnet101(pretrained=pre_trained)

        for p in pretrained_resnet.parameters():
            p.requires_grad = requires_grad

        modules = list(pretrained_resnet.children())[0:layer_cutoff]

        self.model = nn.Sequential(*modules)
        self.model.add_module("Average Pool", nn.AdaptiveAvgPool2d((1, 1)))
        self.model.add_module("Flatten", nn.Flatten())
        # map output layer to output feature size
        # i.e. layer 5 cutoff corresponds to 256 feature size
        in_dim = Thor.resnet_layer_dims[layer_cutoff - 1]
        self.out_dim = in_dim

        if not extract_features:
            for i in range(num_blocks):
                self.model.add_module("Block" + str(i), SeaNormaBlock(in_dim, hidden_size))
                in_dim = hidden_size

            self.model.add_module("Logits", nn.Linear(in_dim, num_classes))
            self.out_dim = num_classes

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def get_out_dim(self):
        return self.out_dim


class IronMan(nn.Module):
    inception_layer_dims = [32, 32, 64, 64, 80, 192, 192, 256, 288, 288, 768, 768, 768, 768, 768, None, 1280, 2048,
                            2048, 2048, 2048, 1000]

    inception_layer_dims_no_aux = [32, 32, 64, 64, 80, 192, 192, 256, 288, 288, 768, 768, 768, 768, 768, 1280, 2048,
                                   2048, 2048, 2048, 1000]

    def __init__(self, num_classes=200, hidden_size=2048, num_blocks=1, requires_grad=False, layer_cutoff=20,
                 use_aux=False, extract_features=False, pre_trained=True):
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

        super().__init__()

        pretrained_inception = torchvision.models.inception_v3(pretrained=pre_trained, aux_logits=use_aux)

        for p in pretrained_inception.parameters():
            p.requires_grad = requires_grad

        modules = list(pretrained_inception.children())[0:layer_cutoff]

        self.model = nn.Sequential(*modules)
        self.model.add_module("Average Pool", nn.AdaptiveAvgPool2d((1, 1)))
        self.model.add_module("Flatten", nn.Flatten())
        # map output layer to output feature size
        # i.e. layer 5 cutoff corresponds to 256 feature size

        if use_aux:
            in_dim = IronMan.inception_layer_dims[layer_cutoff - 1]
        else:
            in_dim = IronMan.inception_layer_dims_no_aux[layer_cutoff - 1]

        self.out_dim = in_dim

        if not extract_features:
            for i in range(num_blocks):
                self.model.add_module("Block" + str(i), SeaNormaBlock(in_dim, hidden_size))
                in_dim = hidden_size

            self.model.add_module("Logits", nn.Linear(in_dim, num_classes))
            self.out_dim = num_classes

    def forward(self, x):
        x = self.model.forward(x)
        return x

    def get_out_dim(self):
        return self.out_dim

class Ensemble(nn.Module):
    # Ensemble model solely used for inference
    def __init__(self, models):
        self.models = models
    
    def forward(self, x):
        outputs = [F.softmax(model(x), dim=1) for model in self.models]
        prob = sum(outputs) / len(outputs)

        return prob


class ConvengersCat(nn.Module):
    # concatenate then FCC

    def __init__(self, num_classes=200, requires_grad=False, pre_trained=True):
        super().__init__()

        self.thor = Thor(extract_features=True, requires_grad=requires_grad, pre_trained=pre_trained)
        self.ironman = IronMan(extract_features=True, requires_grad=requires_grad, pre_trained=pre_trained)
        self.captainamerica = CaptainAmerica(extract_features=True, requires_grad=requires_grad, layer_cutoff=-2, pre_trained=pre_trained)
        
        in_dim = self.thor.get_out_dim() + self.ironman.get_out_dim() + self.captainamerica.get_out_dim()
        
        self.teamup = nn.Sequential()
        self.teamup.add_module("Block1", SeaNormaBlock(in_dim, 4096))
        self.teamup.add_module("Block2", SeaNormaBlock(4096, 2048))
        self.teamup.add_module("Block3", SeaNormaBlock(2048, 1024))
        self.teamup.add_module("FC", nn.Linear(1024, num_classes))

    def forward(self, x):
        thor_out = self.thor.forward(x)
        ironman_out = self.ironman.forward(x)
        captainamerica_out = self.captainamerica.forward(x)
        concat_out = torch.cat((thor_out, ironman_out, captainamerica_out), dim=1)
        out = self.teamup.forward(concat_out)
        return out
    
    def thor_grad(self, requires_grad):
        for p in self.thor.parameters():
            p.requires_grad = requires_grad
    
    def ironman_grad(self, requires_grad):
        for p in self.ironman.parameters():
            p.requires_grad = requires_grad
            
    def captainamerica_grad(self, requires_grad):
        for p in self.ironman.parameters():
            p.requires_grad = requires_grad
            
    def teamup_grad(self, requires_grad):
        for p in self.teamup.parameters():
            p.requires_grad = requires_grad
        
    def end_to_end_grad(self, requires_grad):
        # sets all parameters to require grad
        
        self.thor_grad(requires_grad)
        self.ironman_grad(requires_grad)
        self.captainamerica_grad(requires_grad)
        self.teamup_grad(requires_grad)
               
   
        
