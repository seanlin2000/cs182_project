import torch

class NickFury(object):
    """
    NickFury manages all our convengers
    """
    def __init__(self, model, dataloaders):
        
        self.model = model
        self.trainLoader = dataloaders["train"]
        self.valLoader = dataloaders["val"]
    
    def top_k_accuracy(self, k):
        
        for idx, (images, labels) in enumerate(self.valLoader):
            scores = self.model(images)
            top_k_scores, top_k_labels = torch.topk(scores, k)
            
        