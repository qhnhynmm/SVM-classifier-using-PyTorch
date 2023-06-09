import torch
import torch.nn as nn
import torch.optim as optim
import os

from data_loader.loaddata import LoadData
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
from utils.builder import build_model,build_loss_fn

class Classify_Task:
    def __init__(self, config):
        self.num_epochs = config.num_epochs
        self.patience = config.patience
        self.train_path = config.train_path
        self.valid_path = config.valid_path
        self.test_path = config.test_path
        self.learning_rate = config.learning_rate
        self.save_path = config.save_path
        self.dataloader = LoadData(config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.base_model = build_model(config).to(self.device)
        self.loss_function = build_loss_fn(config)

    def training(self):
        if not os.path.exists(self.save_path):
          os.makedirs(self.save_path)

        train = self.dataloader.load_data(data_path=self.train_path)
        valid = self.dataloader.load_data(data_path=self.valid_path)

        optimizer = optim.Adam(self.base_model.parameters(), lr=self.learning_rate)
        if os.path.exists(os.path.join(self.save_path, 'last_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'last_model.pth'))
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('loaded the last saved model!!!')
            initial_epoch = checkpoint['epoch'] + 1
            print(f"continue training from epoch {initial_epoch}")
        else:
            initial_epoch = 0
            print("first time training!!!")
            train_loss = 0.
            valid_loss = 0.

        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'))
            best_valid_acc = checkpoint['valid_acc']
        else:
            best_valid_acc = 0.
            
        threshold=0
        self.base_model.train()
        for epoch in range(initial_epoch, self.num_epochs + initial_epoch):
            valid_acc = 0.
            train_acc = 0.
            train_loss = 0.
            valid_loss = 0.
            for images, labels in train:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                output = self.base_model(images)
                loss = self.loss_function(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += (output.argmax(1) == labels).sum().item() / labels.size(0)

            with torch.no_grad():
                for images, labels in valid:
                    images, labels = images.to(self.device), labels.to(self.device)
                    output = self.base_model(images)

                    loss = self.loss_function(output, labels)
                    valid_loss += loss.item()
                    valid_acc += (output.argmax(1) == labels).sum().item() / labels.size(0)

            train_loss /= len(train)
            train_acc /= len(train)
            valid_loss /= len(valid)
            valid_acc /= len(valid)

            print(f"epoch {epoch + 1}/{self.num_epochs + initial_epoch}")
            print(f"train loss: {train_loss:.4f} train acc: {train_acc:.4f}")
            print(f"valid loss: {valid_loss:.4f} valid acc: {valid_acc:.4f}")

            # save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'valid_acc': valid_acc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'valid_loss': valid_loss}, os.path.join(self.save_path, 'last_model.pth'))

            # save the best model

            if epoch > 0 and valid_acc < best_valid_acc:
              threshold += 1
            else:
              threshold = 0

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.base_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'valid_acc': valid_acc,
                    'train_acc':train_acc,
                    'train_loss':train_loss,
                    'valid_loss':valid_loss}, os.path.join(self.save_path, 'best_model.pth'))
                print(f"saved the best model with validation accuracy of {valid_acc:.4f}")
            
            # early stopping
            if threshold >= self.patience:
                print(f"early stopping after epoch {epoch + 1}")
                break

    def evaluate(self):
        test_data = self.dataloader.load_test_data( data_path= self.test_path)
        if os.path.exists(os.path.join(self.save_path, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(self.save_path, 'best_model.pth'), map_location=self.device)
            self.base_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print('chưa train model mà đòi test hả')
        self.base_model.eval()

        test_acc = 0.
        true_labels = []
        pred_labels = []
        with torch.no_grad():
            for images, labels in test_data:
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.base_model(images)

                test_acc += (output.argmax(1) == labels).sum().item() / labels.size(0)
                true_labels.extend(labels.cpu().numpy())
                pred_labels.extend(output.argmax(1).cpu().numpy())
        test_acc /= len(test_data)
        print('test accuracy: {:.4f}'.format(test_acc))
        
        f1 = f1_score(true_labels, pred_labels, average='macro')
        print('test F1 score: {:.4f}'.format(f1))
        
        cm = confusion_matrix(true_labels, pred_labels)
        print('confusion matrix:')
        print(cm)