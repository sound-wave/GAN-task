import torch
import torch.nn as nn
import torch.optim as optim



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

PATH = "./models/model.pt"

torch.save(net.state_dict(), PATH)

# Load
device = torch.device('cpu')
model = Net()
model.load_state_dict(torch.load(PATH, map_location=device))
torch.save(net.state_dict(), PATH)

# Load
device = torch.device("cuda")
model = Net()
model.load_state_dict(torch.load(PATH))
model.to(device)


# Save
torch.save(net.state_dict(), PATH)

# Load
device = torch.device("cuda")
model = Net()
# Choose whatever GPU device number you want
model.load_state_dict(torch.load(PATH, map_location="cuda:0"))
# Make sure to call input = input.to(device) on any input tensors that you feed to the model
model.to(device)


# Save
torch.save(net.module.state_dict(), PATH)