import torch
import time

class NickFury(object):
    """
    NickFury manages all our convengers
    """
    def __init__(self, model, dataloaders, datasizes):
        
        self.model = model
        
        self.dataLoader = dict()
        self.dataSize = dict()
        self.dataLoader["train"] = dataloaders["train"]
        self.dataSize["train"] = datasizes["train"]
        self.dataLoader["val"] = dataloaders["val"]
        self.dataSize["val"] = datasizes["val"]
    
    def train(self, optimizer, criterion, lr_scheduler, device, num_epochs=25):
        
        trainLoader = self.dataLoader["train"]
        
        loss_history = []
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print("Epoch {0}:".format(epoch))
            self.model.train() ## put model in training mode
            
            running_loss = 0
            num_points = 0
            num_hits = 0
            ##iterate through one epoch of training data
            for idx, (images, labels) in enumerate(trainLoader):
                
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                scores = self.model(images)
                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                num_points += len(labels)
                num_hits += torch.sum(labels == torch.argmax(scores, dim=1)).item()
                
                print("\r",end='')
                print("Num points is ", num_points, " num hits is ", num_hits)
                print("Training {0:0.2f}%, loss: {1:0.3f}, Accuracy: {2:0.2f}%".format(100*idx/len(trainLoader), running_loss/num_points, 100*num_hits/num_points), end='')
            
            with torch.no_grad():
                per_point_loss = running_loss/self.dataSize["train"]
                train_accuracy = self.accuracy("train")
                self.model.eval()  #put model in evaluation mode to calculate validation
                val_accuracy = self.accuracy("val")
                
            epoch_end_time = time.time()
            hours, rem = divmod(epoch_end_time-epoch_start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print()
            print("Epoch {} completed with elapsed time {:0>2}:{:0>2}:{:05.2f}".format(epoch, int(hours),int(minutes),seconds))
            
            print("Train Loss: {0:0.3f}".format(per_point_loss))
            print("Train Accuracy: {0:.3f}".format(train_accuracy))
            print("Validation Accuracy: {0:.3f}".format(val_accuracy))
                                                
            loss_history.append(per_point_loss)
                                                
        torch.save({
            'net': model.state_dict(),
        }, 'latest.pt')
        
        return loss_history
    
    def accuracy(self, phase):
        return self.top_k_accuracy(1, phase)
        
    def top_k_accuracy(self, k, phase):
        
        total_hits = 0
        for images, labels in self.dataLoader[phase]:
            images = images.to(device)
            labels = labels.to(device)
            scores = self.model(images)
            top_k_scores, top_k_labels = torch.topk(scores, k)
            hits = (labels == top_k_indices.T).any(axis=0)
            total_hits += torch.sum(hits)
            
        return total_hits/self.dataSize[phase]
            
            
            
        