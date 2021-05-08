import torch
import time

class NickFury(object):
    """
    NickFury manages all our convengers
    """
    def __init__(self, model_name, model, dataloaders, datasizes, device):
        
        self.model_name = model_name
        self.model = model
        self.device = device
        self.dataLoader = dict()
        self.dataSize = dict()
        self.dataLoader["train"] = dataloaders["train"]
        self.dataSize["train"] = datasizes["train"]
        self.dataLoader["val"] = dataloaders["val"]
        self.dataSize["val"] = datasizes["val"]

    
    def train(self, optimizer, criterion, lr_scheduler=None, num_epochs=25):
        
        trainLoader = self.dataLoader["train"]
        
        loss_history = []
        best_val_accuracy = -1
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print("Epoch {0}:".format(epoch))
            self.model.train() ## put model in training mode
            
            running_loss = 0
            num_points = 0
            num_hits = 0
            ##iterate through one epoch of training data
            for idx, (images, labels) in enumerate(trainLoader):
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                scores = self.model(images)
                loss = criterion(scores, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                num_points += len(labels)
                num_hits += torch.sum(labels == torch.argmax(scores, dim=1)).item()
                
                # print("\r",end='')
                #print("Num points is ", num_points, " num hits is ", num_hits)
                print("\rTraining {0:0.2f}%, loss: {1:0.3f}, Accuracy: {2:0.2f}%".format(100*idx/len(trainLoader), running_loss/num_points, 100*num_hits/num_points), end='')

            per_point_loss = running_loss / self.dataSize["train"]
                
            epoch_end_time = time.time()
            hours, rem = divmod(epoch_end_time-epoch_start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print()
            print("Epoch {} completed with elapsed time {:0>2}:{:0>2}:{:05.2f}".format(epoch, int(hours),int(minutes),seconds))
            
            # print("Train Loss: {0:0.3f}".format(per_point_loss))
            # print("Train Accuracy: {0:.3f}".format(train_accuracy))
            with torch.no_grad():
                # per_point_loss = running_loss/self.dataSize["train"]
                # train_accuracy = self.accuracy("train")
                self.model.eval()  #put model in evaluation mode to calculate validation
                val_accuracy = self.accuracy("val")
                print("Validation Accuracy: {0:.3f}".format(val_accuracy))
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save({
                        'overnight': self.model.state_dict(),
                    }, self.model_name + '.pt')
                    
            if lr_scheduler:
                lr_scheduler.step()
        
        return loss_history

    def save_model(self, name, filename):
        torch.save({
            name: self.model.state_dict(),
        }, filename)

    def accuracy(self, phase):
        return self.top_k_accuracy(1, phase)
        
    def top_k_accuracy(self, k, phase):
        
        total_hits = 0
        for images, labels in self.dataLoader[phase]:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            scores = self.model(images)
            top_k_scores, top_k_indices = torch.topk(scores, k)
            hits = (labels == top_k_indices.T).any(axis=0)
            total_hits += torch.sum(hits)
            
        return total_hits/self.dataSize[phase]
            
            
            
        