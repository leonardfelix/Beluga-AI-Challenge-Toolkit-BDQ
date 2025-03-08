import random
import torch
import numpy as np
from collections import deque
import pickle

class PrioritisedReplayMemory:
    def __init__(self, maxlen, alpha=0.6):
        self.len = maxlen
        self.alpha = alpha
        self.memory = deque([], maxlen=maxlen)
        self.priorities = deque([], maxlen=maxlen)
        self.epsilon = 1e-5  # Small constant to avoid zero priority
    
    def append(self, transition):
        max_priority = max(self.priorities, default=1.0)
        self.memory.append(transition)
        self.priorities.append(max_priority)
    
    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]
        
        # Compute importance-sampling weights
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        # weights /= weights.max()  # Normalize
        
        states, actions, next_states, rewards, dones, new_state_pddls = zip(*samples)
        
        return [states, actions, next_states, rewards, dones, weights, indices, new_state_pddls]

    def update_priorities(self, indices, td_errors):
        for i, error in zip(indices, td_errors):
            self.priorities[i] = abs(error.item()) + self.epsilon  # Avoid zero priority

    def save_memory_to_file(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.memory, file)

    def load_memory_from_file(self, filename):
        with open(filename, 'rb') as file:
            self.memory = pickle.load(file)
    
    def __len__(self):
        return len(self.memory)