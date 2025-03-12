import torch
import torch.nn.functional as F
import numpy as np
from math import exp

from DQN import DuelingDQN
from prioritised_experience_replay import PrioritisedReplayMemory
import itertools

import yaml
import random

import matplotlib
import matplotlib.pyplot as plt

from datetime import datetime, timedelta
import os
import argparse

from RL_utils import generate_applicable_actions, jig_mask, destination_mask, get_valid_destination, extract_from_actions

DATE_FORMAT = "%m-%d %H:%M:%S"

# directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# use matplotlib agg
matplotlib.use('Agg')

# Large negative number for masking
LARGE_NEG = -float('inf')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class Agent:
    '''
    This is a model constructed using both the Beluga Gym Domain and PDDL domain

    PDDL is used to analyse the states and make an action, with the ability of masking applicable actions. Then setting the next state
    BelugaGym is used to calculate the reward and termination of an action.

    '''

    def __init__(self, hyperparameters, domain=None, gym_compatible_domain=None):
        """
        Initialises the agent with the specified hyperparameters.
        """
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparameters = yaml.safe_load(file)
            instance_hyperparameters = all_hyperparameters[hyperparameters]

        self.hyperparameters = hyperparameters
        self.checkpoint = instance_hyperparameters['checkpoint']
        self.prior_data = instance_hyperparameters['prior_data']
        self.replay_memory_size = instance_hyperparameters['replay_memory_size']
        self.batch_size = instance_hyperparameters['batch_size']
        self.epsilon_start = instance_hyperparameters['epsilon_start']
        self.epsilon_decay = instance_hyperparameters['epsilon_decay']
        self.epsilon_end = instance_hyperparameters['epsilon_end']
        self.network_sync_rate = instance_hyperparameters['network_sync_rate']
        self.discount_factor = instance_hyperparameters['discount_factor']
        self.lr = instance_hyperparameters['lr']
        self.hidden_dims = instance_hyperparameters['hidden_dims']
        self.maximum_simulation_steps = instance_hyperparameters['maximum_simulation_steps']
        self.enable_double_DQN = instance_hyperparameters['enable_double_DQN']
        self.enable_dueling_DQN = instance_hyperparameters['enable_dueling_DQN']
        self.alpha = instance_hyperparameters['alpha']
        self.beta = instance_hyperparameters['beta']
        self.beta_increment = instance_hyperparameters['beta_increment']

        # define loss and optimiser
        self.loss_fn = torch.nn.MSELoss()
        self.optimiser = None
        
        # path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.pth")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.png")
        self.DATA_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.pkl")

        self.domain = domain
        self.gym_compatible_domain = gym_compatible_domain

    def run(self, is_training=True):
        """
        Runs the agent in either training or evaluation mode.
        """
        if is_training:
            start_time = datetime.now()
            last_graph_update = start_time

            log_message = f"Training {start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, "w") as file:
                file.write(log_message)

        # set up the environment
        domain = self.domain
        gym_compatible_domain = self.gym_compatible_domain
        state = domain.reset()

        # get the output and number of output for each branch (jig, destination)
        jigs_ids = [domain.task.objects.index(obj) for obj in domain.task.objects if obj.startswith("jig")]
        
        destination_ids = get_valid_destination(domain, state) # return list of destination

        # get the state and action brach shape
        num_states = len(gym_compatible_domain.make_state_array(state, jigs_ids, destination_ids))
        num_jigs = len(jigs_ids)
        num_destination = len(destination_ids)

        base_reward = 10
        rewards_per_episode = []
        
        # Initialize the policy DQN with the specified dimensions and move it to the appropriate device
        policy_dqn = DuelingDQN(state_dim=num_states, jig_dim=num_jigs, destination_dim=num_destination, hidden_dim=self.hidden_dims).to(device)

        # Set up the optimizer with the policy DQN parameters and learning rate
        self.optimiser = torch.optim.Adam(params=policy_dqn.parameters(), lr=self.lr, betas=(0.9, 0.999))

        if is_training:

            # Print confirmation conficguration
            # if provided checkpoint, load from it
            if self.checkpoint != "":
                print("Loading from checkpoint...")
                policy_dqn.load_state_dict(torch.load(self.checkpoint))
                print(f"Checkpoint loaded from {self.checkpoint}")
            else:
                print("Training from scratch...")

            if self.enable_double_DQN:
                print("Double DQN enabled")
                
            # set up the memory, target network, and step count
            # memory = ReplayMemory(self.replay_memory_size)

            memory = PrioritisedReplayMemory(maxlen=self.replay_memory_size, alpha=self.alpha)
            beta = self.beta  # Importance sampling weight

            # load if prior data is provided
            if self.prior_data != "":
                print("Loading prior data...")
                memory.load_memory_from_file(self.prior_data)
                print(f"Prior data loaded from {self.prior_data}")


            epsilon = self.epsilon_start

            target_dqn = DuelingDQN(state_dim=num_states, jig_dim=num_jigs, destination_dim=num_destination, hidden_dim=self.hidden_dims).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())

            epsilon_history = []
            self.loss_history = []
            
            step_count = 0

            best_reward = -9999999
            

        else:
            # test the model to evaluation mode
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()


        # run the episodes
        for episode in itertools.count():
            state_pddl = domain.reset()
            
            terminated = False
            episode_reward = 0.0
            simulation_step = 0

            # list to keep track of the state visited for reward calculations
            states_visited = []

            while (not terminated and simulation_step < self.maximum_simulation_steps):
    
                state = gym_compatible_domain.make_state_array(state_pddl, jigs_ids, destination_ids)
                state = torch.tensor(state, dtype=torch.float32).to(device)

                states_visited.append(state_pddl)

                # case where the option of change beluga is present choose it and continue
                available_actions = domain.get_applicable_actions(state_pddl)
                

                beluga_complete = any(action.action_id == 8 for action in available_actions._elements)
                if beluga_complete:
                    action = next(action for action in available_actions._elements if action.action_id == 8)
                    o = domain.step(action)
                    new_state_pddl = o.observation
                    terminated = o.termination
                    reward = self.get_reward(states_visited, new_state_pddl, action.action_id, terminated) 
                    state_pddl = new_state_pddl

                    # log action
                    print(f"Action: {action} Reward: {reward}")
                    continue


                # Next action using epsilon-greedy
                if is_training and random.random() < epsilon:
                    action = available_actions.sample()
                    
                    jig_id_obj, action_id, destination_id_obj = extract_from_actions(action)

                    # convert from object index to raw index (start from 0)
                    jig_id = jigs_ids.index(jig_id_obj)
                    destination_id = destination_ids.index(destination_id_obj)
                    

                else:
                    with torch.no_grad():
                        destination = policy_dqn(state.unsqueeze(0))   #this output jig_id, action_id, and destination_id from 0

                        destination_m = destination_mask(domain, state_pddl, destination_ids)
                        destination_id = (destination.masked_fill(~destination_m, LARGE_NEG)).argmax().item()
                        destination_id_obj = destination_ids[destination_id]

                        jig_m = jig_mask(destination_id_obj, domain, state_pddl)

                        # Get indices where the value is 1
                        valid_indices = torch.nonzero(jig_m, as_tuple=True)[0]
                        jig_id = valid_indices[torch.randint(0, valid_indices.shape[0], (1,))].item()

                        # jig_id = (jig.masked_fill(~jig_m, LARGE_NEG)).argmax().item()
                        jig_id_obj = jigs_ids[jig_id]
                    
                        action = generate_applicable_actions(jig_id_obj, destination_id_obj, domain, state_pddl)

                # apply action
                if action is None: # no applicable action
                    new_state = state_pddl
                    reward = -1
                    terminated = True
                else:
                    o = domain.step(action)

                    new_state_pddl = o.observation
                    terminated = o.termination
                    reward = self.get_reward(states_visited, new_state_pddl, action.action_id, terminated) 

                    # # set hard limit to state repetition
                    # if reward == -5:
                    #     terminated = True

                episode_reward += reward

                # convert new_state and reward to tensor
                new_state = torch.tensor(gym_compatible_domain.make_state_array(new_state_pddl, jigs_ids, destination_ids), dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)

                # log action
                print(f"Action: {action} Reward: {reward}")

                if is_training:
                    # append to memory
                    memory.append((state, [jig_id, destination_id], new_state, reward, terminated, new_state_pddl))
                    step_count += 1

                state_pddl = new_state_pddl
                simulation_step += 1

            rewards_per_episode.append(episode_reward)
            print("episode terminated")

            # save model when best rewards is obtained to log  -----------------------------------
            if is_training:
                if episode_reward > best_reward:
                    log_message = f"Training {datetime.now().strftime(DATE_FORMAT)}: New best reward: {best_reward} Time taken: {datetime.now() - start_time} Episode: {episode} Saved memory size: {len(memory)} out of {self.replay_memory_size}\n"
                    print(log_message)
                    with open(self.LOG_FILE, "a") as file:
                        file.write(log_message)

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward

                    # # write into a file the memory contents
                    # memory.save_memory_to_file(self.DATA_FILE)


                #update graph every 10 second
                current_time = datetime.now()
                if current_time - last_graph_update > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history, self.loss_history)
                    last_graph_update = current_time


                # exponential decay for epsilon
                epsilon = max(self.epsilon_end, self.epsilon_decay * epsilon)
                epsilon_history.append(epsilon)

                # linear anneal for beta
                beta = min(1.0, beta + self.beta_increment)

                if len(memory) > self.batch_size:
                    batch = memory.sample(self.batch_size, beta)

                    self.optimise(policy_dqn, target_dqn, batch, memory, destination_ids, domain)

                    # sync the target network with respect to the policy network
                    if step_count > self.network_sync_rate:
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

    def optimise(self, policy_dqn, target_dqn, batch, memory, destination_ids, domain):
        """
        Optimises the policy DQN using the provided batch.
        """
        # optimise the network
        # Note: 
        # - actions contains [jig_id, destination_id]. 
        states, actions, new_states, rewards, terminated, weights, indices, new_state_pddls = batch   

        batch_size = len(states)  # Number of batch size

        destination_ids_tensor = torch.tensor(destination_ids, dtype=torch.int64).to(device)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        indices = torch.tensor(indices, dtype=torch.int64).to(device)


        # Pre-allocate memory using torch.empty
        best_jig = torch.empty(batch_size, dtype=torch.int64)
        best_destination = torch.empty(batch_size, dtype=torch.int64)

        with torch.no_grad():
            # Compute next Q-values for all states  NON DUELING_DQN        
            Q_next_destination = target_dqn(new_states)

            # Compute best destinations
            destination_masks = [destination_mask(domain, new_state_pddl, destination_ids) for new_state_pddl in new_state_pddls]
            best_destination = torch.argmax(Q_next_destination.masked_fill(~torch.stack(destination_masks), LARGE_NEG), dim=1)
            # best_destination_obj = destination_ids_tensor[best_destination]


            # Compute best Jigs
            # jig_masks = [jig_mask(destination_id, domain, new_state_pddl) for destination_id, new_state_pddl in zip(best_destination_obj, new_state_pddls)]
            # best_jig = torch.argmax(Q_next_jig.masked_fill(~torch.stack(jig_masks), LARGE_NEG), dim=1)

            
            # target_Q_jig = Q_next_jig.gather(1, best_jig.unsqueeze(1)).squeeze(1)
            target_Q_destination = Q_next_destination.gather(1, best_destination.unsqueeze(1)).squeeze(1)

            # # Use target network to evaluate best actions (Duelling DQN)
            # Q_target_jig, Q_target_destination = target_dqn(new_states)
            # target_Q_jig = Q_target_jig.gather(1, best_jig.unsqueeze(1)).squeeze(1)
            # target_Q_destination = Q_target_destination.gather(1, best_destination.unsqueeze(1)).squeeze(1)

            # Compute Bellman targets
            # target_Q_jig = rewards + self.discount_factor * target_Q_jig * (1 - terminated)
            target_Q_destination = rewards + self.discount_factor * target_Q_destination * (1 - terminated)

        # Get current state Q value
        Q_destination = policy_dqn(states)
        # Q_jig = Q_jig.gather(1, actions[:, 0].unsqueeze(1)).squeeze(1)
        Q_destination = Q_destination.gather(1, actions[:, 1].unsqueeze(1)).squeeze(1)


       # Compute MSE loss per branch
        # loss_jig = self.loss_fn(Q_jig, target_Q_jig)
        td_errors = target_Q_destination - Q_destination

        # Compute final loss as mean across branches
        loss = torch.mean(td_errors**2 * weights)

        # Optimize network
        self.optimiser.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 10.0)

        self.optimiser.step()

        self.loss_history.append(loss.item())

        # TD error for prioritised replay updates
        memory.update_priorities(indices, td_errors)


    def save_graph(self, rewards_per_episode, epsilon_history, loss_history):
        
        fig = plt.figure(1)

        # plot rewards mean

        plt.subplot(1,2,1)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(rewards_per_episode)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-100):x+1])

        plt.plot(mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per episode")

        # plt.plot(loss_history)
        # plt.xlabel("Loss")
        # plt.ylabel("Reward")
        # plt.title("Loss per episode")



        # plot epsilon
        plt.subplot(1,2,2)
        plt.plot(epsilon_history)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon per episode")

        fig.savefig(self.GRAPH_FILE)

        plt.close(fig)

    def get_reward(self, states_visited, new_state, action_id, terminated):
        # give reward for the doing actions of unloading beluga, loading beluga, and sending to hangar 
        if terminated:
            return 5
        elif action_id in [0,1]:
            return 1
        elif action_id in [3]:
            return 2
        else:
            return -5 if new_state in states_visited else -(1/self.maximum_simulation_steps) 


if __name__ == "__main__":
    # parse command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument("gym_environment", help="Name of the gym environment")
    parser.add_argument("--train", help="Train the agent", action="store_true")
    parser.add_argument("--render", help="Render the environment", action="store_true")
    args = parser.parse_args()
    
    dql = Agent(args.gym_environment)
    if args.train:
        dql.run(is_training=True, render=args.render)
    else:
        dql.run(is_training=False, render=args.render)  