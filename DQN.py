import torch
from torch import nn
import torch.nn.functional as F

def scale_grad_hook(grad):
    return grad * (1 / 3)  # Scale gradients before entering shared layers


class DuelingDQN(nn.Module):

    def __init__(self, state_dim, jig_dim, destination_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim*2)
        self.bn1 = nn.BatchNorm1d(hidden_dim*2)

        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        # self.bn3 = nn.BatchNorm1d(hidden_dim)

        # self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn4 = nn.BatchNorm1d(hidden_dim)

        # self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn5 = nn.BatchNorm1d(hidden_dim)

        # self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        # self.bn6 = nn.BatchNorm1d(hidden_dim)

        # Value network
        self.fc_value = nn.Linear(hidden_dim, hidden_dim//2)
        self.value = nn.Linear(hidden_dim//2, 1)

        # # Advantage network
        # self.fc_jig_advantage = nn.Linear(hidden_dim, hidden_dim//2)
        # self.jig_advantage = nn.Linear(hidden_dim//2, jig_dim)

        self.fc_destination_advantage = nn.Linear(hidden_dim, hidden_dim//2)
        self.destination_advantage = nn.Linear(hidden_dim//2, destination_dim)

        # # Register hook to scale gradients
        # self.fc1.weight.register_hook(scale_grad_hook)
        # self.fc2.weight.register_hook(scale_grad_hook)
        # self.fc3.weight.register_hook(scale_grad_hook)
        # self.fc4.weight.register_hook(scale_grad_hook)
        # self.fc5.weight.register_hook(scale_grad_hook)
        # self.fc6.weight.register_hook(scale_grad_hook)

        self.init_weights()

    def forward(self, x):
        if x.shape[0] > 1:  # Only apply BatchNorm if training for batch
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            # x = F.relu(self.bn3(self.fc3(x)))
            # x = F.relu(self.bn4(self.fc4(x)))
            # x = F.relu(self.bn5(self.fc5(x)))
            # x = F.relu(self.bn6(self.fc6(x)))
        else:
            x = F.relu(self.fc1(x))  # Skip BatchNorm for single input
            x = F.relu(self.fc2(x))
            # x = F.relu(self.fc3(x))
            # x = F.relu(self.fc4(x))
            # x = F.relu(self.fc5(x))
            # x = F.relu(self.fc6(x))
        
        # Compute the value function
        v = F.relu(self.fc_value(x))
        value = self.value(v)

        # # Compute the advantages for each bracnh of action, jig and destination
        # a_jig = F.relu(self.fc_jig_advantage(x))
        # advantage_jig = self.jig_advantage(a_jig)

        a_destination = F.relu(self.fc_destination_advantage(x))
        advantage_destination = self.destination_advantage(a_destination)

        # Compute Q-values for each dimension separately
        # Q_jig = value + (advantage_jig - advantage_jig.mean(dim=1, keepdim=True))
        Q_destination = value + (advantage_destination - advantage_destination.mean(dim=1, keepdim=True))        

        return Q_destination 

    def init_weights(self):     # Xavier initialisation
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight) 
                nn.init.zeros_(m.bias)            

