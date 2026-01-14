import torch
import torchvision
import random

from torch import nn

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        #Set up the layers for the Neural Net:
        """
        1) Take the flattened data of a picture (28*28) and output 512 bits of data
        2) Apply ReLU
        3) Hidden layer does funky magic
        4) Apply ReLU
        5) Final layer outputs its verdict: 0-9
        """
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,10),
        )
    
    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

def training_loop(dataloader, model, loss_fn, optimizer, batch_size=64):
    model.train()
    size = len(dataloader.dataset)

    for batch, (X,y) in enumerate(dataloader):
        #Prediction and compute loss
        pred = model(X)
        loss = loss_fn(pred, y)

        #Backprop
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    #Make sure we dont compute a gradient here
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

class Main():
    def __init__(self):

        #Import training data. To test algorithm, we will be using MNIST database, but in the future I would like to gather my own data to train on.
        train_data = torchvision.datasets.MNIST(
            root="",
            train=True,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        #Importing testing data. 
        test_data = torchvision.datasets.MNIST(
            root="",
            train=False,
            download=True,
            transform=torchvision.transforms.ToTensor()
        )

        self.training_dataloader = torch.utils.data.DataLoader(train_data,batch_size=64)
        self.test_dataloader = torch.utils.data.DataLoader(test_data,batch_size=64)

        self.model = NeuralNet()

        self.learning_rate = 1e-3
        self.epochs = 5

        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):
        for t in range(self.epochs):
            print(f"Epoch {t+1} \n ------------------------")
            training_loop(self.training_dataloader,self.model,self.loss_fn,self.optimizer)
            test_loop(self.test_dataloader,self.model,self.loss_fn)
        
        torch.save(self.model.state_dict(),'model.pth')

    def grab_images(self):
        # Get a random index from the test dataset
        random_idx = random.randint(0, len(self.test_dataloader.dataset) - 1)
        image = self.test_dataloader.dataset[random_idx]
        # image is a tuple of (tensor, label)
        return image[0].unsqueeze(0), image[1]  # Return as single-item batch and label

if __name__ == "__main__":

    main = Main()
    main.train()