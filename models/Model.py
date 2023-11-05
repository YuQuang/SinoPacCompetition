import torch.nn as nn

# Can reference https://blog.csdn.net/ChenVast/article/details/82107490

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(46, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.02, True),
            nn.Linear(512, 2048),
            nn.LeakyReLU(0.02, True),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.02, True),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.model(x)