import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class myCNN(nn.Module):
    def __init__(self):
        super(myCNN, self).__init__()
        #defining the convolutinal layers to be used in classes
        #first convolutional layer with kernel (5x5), stride (1x1) and 32 output channels
        self.convolution1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=0)
        #max pooling (2x2) with stride (2x2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #second convilutional layer with (5x5) kernel, stride (1x1) and 64 output channels
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0)
        #first fully-connected layer with input size being the output size of max pooling layer 1024
        self.fully_connected1 = nn.Linear(64 * 4 * 4, 1024)
        #second fully-connected layer with input size 1024 and output size 256
        self.fully_connected2 = nn.Linear(1024, 256)
        #output layer with nput 256 and output 10 (classes)
        self.output_layer = nn.Linear(256, 10)
        
        # initializing weights with Xavier Uniform
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)

    def forward(self, x):
        #applying relu activation function to the first layer
        x = torch.relu(self.convolution1(x))
        #max pooling
        x = self.pool(x)
        #applying relu activation function to the second convolutional layer
        x = torch.relu(self.convolution2(x))
        #max pooling
        x = self.pool(x)
        #flatten for fully connected layer
        x = x.view(x.size(0), -1)
        #applying relu to the first fully connected layer
        x = torch.relu(self.fully_connected1(x))
        #applying relu to the second fully connected layer
        x = torch.relu(self.fully_connected2(x))
        #output layer
        x = self.output_layer(x)
        return x

def train_and_validate(model, train_loader,validation_loader,criterion,optimizer,number_epochs=30):
    #list to store train losses
    train_losses = []
    #list to store validation losses
    validation_losses = []
    #list to store train accuracies
    train_accuracies = []
    #list to store validation accuracies
    validation_accuracies = []
    
    for epoch in range(number_epochs):
        # training
        model.train()
        correct_train = 0
        total_train = 0
        epoch_training_loss = 0.0
        #train iterations
        for images, labels in train_loader:
            #set the parameter gradients to zero
            optimizer.zero_grad()
            #forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            #backward pass an update
            loss.backward()
            optimizer.step()
            
            #add training loss
            epoch_training_loss += loss.item() * images.size(0)
            #compute accuracy
            tensors, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
        
        train_accuracy = correct_train / total_train
        train_losses.append(epoch_training_loss)
        train_accuracies.append(train_accuracy)
        
        # validation
        model.eval()
        correct_validation = 0
        total_validation = 0
        epoch_validation_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                #only doing thr forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_validation_loss += loss.item() * images.size(0)
                tensors, preds = torch.max(outputs, 1)
                correct_validation += (preds == labels).sum().item()
                total_validation += labels.size(0)
        
        validation_accuracy = correct_validation / total_validation
        validation_losses.append(epoch_validation_loss)
        validation_accuracies.append(validation_accuracy)
        
        print(f"epoch [{epoch}], "
              f"train loss: {epoch_training_loss/total_train}, train accuracy: {train_accuracy}, "
              f"validation loss: {epoch_validation_loss/total_validation}, validation accuracy: {validation_accuracy}")
    
    # save the best model from last epoch
    torch.save(model.state_dict(), "best_cnn.pth")
    
    return train_losses, validation_losses, train_accuracies, validation_accuracies

if __name__ == "__main__":
    #loading the data
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(root=".", train=True, download=True, transform=transform)
    validation_set = torchvision.datasets.FashionMNIST(root=".", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=32, shuffle=False)
    torch.manual_seed(0)
    # defining the model
    model = myCNN()
    #using cross entropy loss
    criterion = nn.CrossEntropyLoss()
    #SGD optimier
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    train_losses, val_losses, train_accuracies, val_accuracies = train_and_validate(model, train_loader, validation_loader, criterion, optimizer, number_epochs=30)
