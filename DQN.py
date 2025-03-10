import torch
from torch import nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):

    def __init__(self, state_dim, action_dim, jig_dim, destination_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim*4)

        self.fc2 = nn.Linear(hidden_dim*4, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, hidden_dim)
        # self.fc4 = nn.Linear(hidden_dim*2, hidden_dim)

        # Value network
        self.fc_value = nn.Linear(hidden_dim, hidden_dim//2)
        self.value = nn.Linear(hidden_dim//2, 1)

        # Advantage network
        self.fc_jig_advantage = nn.Linear(hidden_dim, hidden_dim//2)
        self.jig_advantage = nn.Linear(hidden_dim//2, jig_dim)

        self.fc_action_advantage = nn.Linear(hidden_dim, hidden_dim//2)
        self.action_advantage = nn.Linear(hidden_dim//2, action_dim)

        self.fc_destination_advantage = nn.Linear(hidden_dim, hidden_dim//2)
        self.destination_advantage = nn.Linear(hidden_dim//2, destination_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        
        # Compute the value function
        v = F.relu(self.fc_value(x))
        value = self.value(v)

        # Compute the advantages for each bracnh of action, jig and destination
        a_jig = F.relu(self.fc_jig_advantage(x))
        advantage_jig = self.jig_advantage(a_jig)

        a_action = F.relu(self.fc_action_advantage(x))
        advantage_action = self.action_advantage(a_action)

        a_destination = F.relu(self.fc_destination_advantage(x))
        advantage_destination = self.destination_advantage(a_destination)

        # Compute Q-values for each dimension separately
        Q_jig = value + (advantage_jig - advantage_jig.mean(dim=1, keepdim=True))
        Q_action = value + (advantage_action - advantage_action.mean(dim=1, keepdim=True))
        Q_destination = value + (advantage_destination - advantage_destination.mean(dim=1, keepdim=True))        

        return Q_jig, Q_action, Q_destination 

