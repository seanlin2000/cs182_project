import torch
import time
import random
from advertorch.attacks import LinfPGDAttack
from advertorch.attacks import GradientSignAttack
from advertorch.context import ctx_noparamgrad_and_eval

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
        self.loss_history = []
        self.adversaries = []
    
    def train(self, optimizer, criterion, lr_scheduler=None, num_epochs=25, adv_train=False):
        
        trainLoader = self.dataLoader["train"]
        
        loss_history = []
        best_val_accuracy = -1
        if adv_train:
            self.build_adversaries(self.model, criterion)
            
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            print("Epoch {0}:".format(epoch))
            self.model.train() ## put model in training mode
            
            running_loss = 0
            num_points = 0
            num_hits = 0

            adv_loss = 0
            adv__points = 0
            adv_hits = 0
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
                
                print("\rTraining {0:0.2f}%, loss: {1:0.3f}, Accuracy: {2:0.2f}%".format(100*idx/len(trainLoader), running_loss/num_points, 100*num_hits/num_points), end='')

                # Train on an adversarial minibatch
                if adv_train:
                    adversary = random.choice(self.adversaries)
                    with ctx_noparamgrad_and_eval(self.model):
                        adv_images = adversary.perturb(images, labels)
                    adv_images = adv_images.to(device)
                    optimizer.zero_grad()
                    scores = self.model(images)
                    loss = criterion(scores, labels)
                    loss.backward()
                    optimizer.step()

                    adv_loss += loss.item()
                    adv_points += len(labels)
                    adv_hits += torch.sum(labels == torch.argmax(scores, dim=1)).item()
                    print()
                    print("\rTraining {0:0.2f}%, loss: {1:0.3f}, Accuracy: {2:0.2f}%".format(100*idx/len(trainLoader), adv_loss/adv_points, 100*adv_hits/adv_points), end='')

            per_point_loss = running_loss / self.dataSize["train"]
            loss_history.append(per_point_loss)
            epoch_end_time = time.time()
            hours, rem = divmod(epoch_end_time-epoch_start_time, 3600)
            minutes, seconds = divmod(rem, 60)
            print()
            print("Epoch {} completed with elapsed time {:0>2}:{:0>2}:{:05.2f}".format(epoch, int(hours),int(minutes),seconds))
            
            with torch.no_grad():
                self.model.eval()  #put model in evaluation mode to calculate validation
                val_accuracy = self.accuracy("val")
                print("Validation Accuracy: {0:.3f}".format(val_accuracy))
                if adv_train:
                    print("Adversarial validation accuracy: {0:.3f}".format(self.adversarial_accuracy("val")))
                # print("Per Point Loss: {0:.3f}".format(per_point_loss))
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    torch.save({
                        'overnight': self.model.state_dict(),
                    }, self.model_name + '.pt')
                    
            if lr_scheduler:
                lr_scheduler.step()
                
        self.loss_history.extend(loss_history)
        return loss_history
    
    def save_model(self, name, filename):
        torch.save({
            name: self.model.state_dict(),
        }, filename)
    
    def save_loss_history(self, filename):
        torch.save(self.loss_history_tensor,filename)
        
        
    def get_total_loss_history(self):
        return self.loss_history
    
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

    def adversarial_accuracy(self, phase):
        return self.top_k_accuracy_adversary(1, phase)
    
    def top_k_accuracy_adversary(self, k, phase):

        total_hits = 0
        for images, labels in self.dataLoader[phase]:
            images = images.to(self.device)
            labels = labels.to(self.device)
            adversary = random.choice(self.adversaries)
            adv_images = adversary.perturb(images, labels)
            adv_images = adv_images.to(device)
            
            scores = self.model(adv_images)
            top_k_scores, top_k_indices = torch.topk(scores, k)
            hits = (labels == top_k_indices.T).any(axis=0)
            total_hits += torch.sum(hits)
            
        return total_hits/self.dataSize[phase]


    def build_adversaries(self, model, criterion):
        eps_list = [.05, .1, .15, .2, .25, .3]
        for eps in eps_list:
            pgd_adv = LinfPGDAttack(model, criterion, eps=eps)
            fgsm_adv = GradientSignAttack(model, criterion, eps=eps)
            self.adversaries.append(pgd_adv)
            self.adversaries.append(fgsm_adv)
            
            
            
        