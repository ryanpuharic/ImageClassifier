import torch
import PIL
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

#Get data
train = datasets.MNIST(root = "data", download = True, train = True, transform = ToTensor())
dataset = DataLoader(train, 32)

#Image classifier NN
class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10)
        )
    def forward(self, x):
        return self.model(x)
    
#Instance of NN
clf = ImageClassifier().to('cpu')
opt = Adam(clf.parameters(), lr = 1e-3)
loss_fn = nn.CrossEntropyLoss()

#Classify
if __name__ == "__main__":
    with open('model_state.pt', 'rb') as f:
        clf.load_state_dict(load(f))

    mnist_data = train.data
    mean = np.mean(mnist_data.numpy())
    std = np.std(mnist_data.numpy())

    img = Image.open('./images/img_12.jpg').convert('L')
    img_array = np.array(img)
    img_norm = (img_array - mean) / std
    resized_img = Image.fromarray(img_norm)
    resized_img = resized_img.resize((28, 28))  

    img_tensor = ToTensor()(resized_img).unsqueeze(0).to('cpu')

    print(torch.argmax(clf(img_tensor)))

    # TRAINING NN
    # for epoch in range(10):
    #     for batch in dataset:
    #         X,y = batch
    #         X, y = X.to('cpu'), y.to('cpu')
    #         yhat = clf(X)
    #         loss = loss_fn(yhat, y)

    #         #backprop
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()

    #     print(f"Epoch:{epoch} loss is {loss.item()}")

    # with open('model_state.pt', 'wb') as f:
    #     save(clf.state_dict(), f)
