import torch
import torch.nn.functional as F
import numpy as np
import yaml
import random
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

from RL_utils import generate_applicable_actions, jig_mask, destination_mask, get_valid_destination, extract_from_actions
from evaluation.planner_api import BelugaPlan

from DQN import DuelingDQN
from prioritised_experience_replay import PrioritisedReplayMemory

import wandb

DATE_FORMAT = "%m-%d %H:%M:%S"

# directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')

# Large negative number for masking
LARGE_NEG = -float('inf')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Agent:
    '''
    This is a model constructed using both the SkdPDDLDomain and Beluga Gym Compatible Domain, but not using the Gym API.

    Instead of using the Gym interface, I directly use PDDL state from the SkdPDDLDomain for making an action, with the ability of masking applicable actions.
    The PDDL states is initialy converted to  state array using the function in Beluga Gym Compatible Domain, to be processed by the model.
    The rewards and termination is then calculated based on the visited states and action taken by the model.

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
        self.alpha = instance_hyperparameters['alpha']
        self.beta = instance_hyperparameters['beta']
        self.beta_increment = instance_hyperparameters['beta_increment']
        self.episode_run = instance_hyperparameters['episode_run']

        # define loss and optimiser
        self.optimiser = None
        
        # path to run info
        self.LOG_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.log")
        self.MODEL_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.pth")
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f"{self.hyperparameters}.png")
        
        self.domain = domain
        self.gym_compatible_domain = gym_compatible_domain

        with open(os.path.join(RUNS_DIR, 'hyperparameters.yaml'), 'w') as file: # save hyperparameters
            yaml.dump(all_hyperparameters, file)

        # Log to weights and biases. Start a new wandb run to track this script.
        self.wbrun = wandb.init(
            # Set the wandb project where this run will be logged.
            project="Beluga-DQN",
            # Track hyperparameters and run metadata.
            config={
                "Description": "Default run with 3 jigs",
                **instance_hyperparameters,
            },
        )

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
        destination_ids = get_valid_destination(domain, state)

        # get the state and action brach shape
        num_states = len(gym_compatible_domain.make_state_array(state, jigs_ids, destination_ids))
        num_jigs = len(jigs_ids)
        num_destination = len(destination_ids)

        rewards_per_episode = []
        best_actions = None
        
        # Initialize the policy DQN with the specified dimensions
        policy_dqn = DuelingDQN(state_dim=num_states, jig_dim=num_jigs, destination_dim=num_destination, hidden_dim=self.hidden_dims).to(device)
        self.optimiser = torch.optim.Adam(params=policy_dqn.parameters(), lr=self.lr, betas=(0.9, 0.999))
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimiser, T_max=self.maximum_simulation_steps, eta_min=1e-8)


        if is_training:
            # If provided checkpoint, load from it
            if self.checkpoint != "":
                print("Loading from checkpoint...")
                policy_dqn.load_state_dict(torch.load(self.checkpoint))
                print(f"Checkpoint loaded from {self.checkpoint}")
            else:
                print("Training from scratch...")

            memory = PrioritisedReplayMemory(maxlen=self.replay_memory_size, alpha=self.alpha)

            beta = self.beta  # Importance sampling weight
            epsilon = self.epsilon_start
            epsilon_history = []
            self.loss_history = []
            step_count = 0
            best_reward = -9999999

            target_dqn = DuelingDQN(state_dim=num_states, jig_dim=num_jigs, destination_dim=num_destination, hidden_dim=self.hidden_dims).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            
        else:
            # Test the model to evaluation mode
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()

        # Run the episodes
        for episode in range(self.episode_run):
            state_pddl = domain.reset()
            terminated = False
            episode_reward = 0.0
            simulation_step = 0
            states_visited = []
            repeated_states = 0
            taken_actions = []
            goal_reached = False

            while (not terminated and simulation_step < self.maximum_simulation_steps):
                state = gym_compatible_domain.make_state_array(state_pddl, jigs_ids, destination_ids)
                state = torch.tensor(state, dtype=torch.float32).to(device)
                states_visited.append(state_pddl)

                available_actions = domain.get_applicable_actions(state_pddl)
                
                # Case where the option of change beluga is present choose it and continue
                beluga_complete = any(action.action_id == 8 for action in available_actions._elements)
                if beluga_complete:
                    action = next(action for action in available_actions._elements if action.action_id == 8)
                    o = domain.step(action)
                    new_state_pddl = o.observation
                    terminated = o.termination
                    reward = self.get_reward(states_visited, new_state_pddl, action.action_id, terminated, num_jigs) 
                    state_pddl = new_state_pddl
                    taken_actions.append(action)

                    # log action
                    print(f"Action: {action} Reward: {reward}")
                    continue

                if is_training and random.random() < epsilon:   # Next action using epsilon-greedy
                    action = available_actions.sample()
                    jig_id_obj, action_id, destination_id_obj = extract_from_actions(action)
                    jig_id = jigs_ids.index(jig_id_obj)
                    destination_id = destination_ids.index(destination_id_obj)

                else:   # Next action from the model
                    with torch.no_grad():
                        destination, jig = policy_dqn(state.unsqueeze(0))   # This output destination_id, jig_id from 0 index
                        destination_m = destination_mask(domain, state_pddl, destination_ids)
                        destination_id = (destination.masked_fill(~destination_m, LARGE_NEG)).argmax().item()
                        destination_id_obj = destination_ids[destination_id]

                        jig_m = jig_mask(destination_id_obj, domain, state_pddl)
                        jig_id = (jig.masked_fill(~jig_m, LARGE_NEG)).argmax().item()
                        jig_id_obj = jigs_ids[jig_id]
                    
                        action = generate_applicable_actions(jig_id_obj, destination_id_obj, domain, state_pddl)

                if action is None: # Case of invalid action
                    new_state = state_pddl
                    reward = -1
                    terminated = True
                else:
                    o = domain.step(action)

                    new_state_pddl = o.observation
                    terminated = o.termination
                    reward = self.get_reward(states_visited, new_state_pddl, action.action_id, terminated, num_jigs) 
                    taken_actions.append(action)

                    # check if goal is reached
                    if terminated:
                        goal_reached = True

                    # set soft limit to state repetition cycles (terminate the episode after more than num_jigs*2 consecutive repetitions)
                    if reward == -10:
                        if repeated_states > (num_jigs)*2:
                            terminated = True
                        else:
                            repeated_states += 1
                    else:
                        repeated_states = 0

                episode_reward += reward
                new_state = torch.tensor(gym_compatible_domain.make_state_array(new_state_pddl, jigs_ids, destination_ids), dtype=torch.float32).to(device)
                reward = torch.tensor(reward, dtype=torch.float32).to(device)

                # Log action
                print(f"Action: {action} Reward: {reward}")

                if is_training: # Append to memory
                    memory.append((state, [jig_id, destination_id], new_state, reward, terminated, new_state_pddl))
                    step_count += 1

                state_pddl = new_state_pddl
                simulation_step += 1

            rewards_per_episode.append(episode_reward)
            print("episode terminated") # Log termination

            if is_training:
                if best_actions is None:    # Initialize best_actions to not be None
                    best_actions = tuple(taken_actions)
                if episode_reward > best_reward and goal_reached:   # Save model when best rewards is obtained and goal is reached
                    log_message = f"Training {datetime.now().strftime(DATE_FORMAT)}: New best reward: {best_reward} Time taken: {datetime.now() - start_time} Episode: {episode} Saved memory size: {len(memory)} out of {self.replay_memory_size}\n"
                    print(log_message)
                    with open(self.LOG_FILE, "a") as file:
                        file.write(log_message)

                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    best_reward = episode_reward
                    best_actions = tuple(taken_actions)

                # Update graph every 10 second
                current_time = datetime.now()
                if current_time - last_graph_update > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history, self.loss_history)
                    last_graph_update = current_time

                # Exponential decay for epsilon
                epsilon = max(self.epsilon_end, self.epsilon_decay * epsilon)
                epsilon_history.append(epsilon)

                # Linear anneal for beta
                beta = min(1.0, beta + self.beta_increment)

                # Log metrics to wandb.
                self.wbrun.log({"Reward": episode_reward, "epsilon": epsilon})

                if len(memory) > self.batch_size:
                    batch = memory.sample(self.batch_size, beta)
                    self.optimise(policy_dqn, target_dqn, batch, memory, destination_ids, domain)

                    if step_count > self.network_sync_rate: # Sync the target network with respect to the policy network
                        target_dqn.load_state_dict(policy_dqn.state_dict())
                        step_count = 0

        # Log training output
        log_message = f"Training done in: {datetime.now() - start_time} Best reward: {best_reward} Best actions:\n"
        print(log_message)
        with open(self.LOG_FILE, "a") as file:
            file.write(log_message)

        for a in best_actions:
            print(a)
            with open(self.LOG_FILE, "a") as file:
                file.write(str(a) + "\n")

        return best_actions, best_reward

    def optimise(self, policy_dqn, target_dqn, batch, memory, destination_ids, domain):
        """
        Optimises the policy DQN using the provided batch.

        Note: 
        - actions contains [destination_id, jig_id]. 
        """
        states, actions, new_states, rewards, terminated, weights, indices, new_state_pddls = batch   

        destination_ids_tensor = torch.tensor(destination_ids, dtype=torch.int64).to(device)

        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        terminated = torch.tensor(terminated, dtype=torch.float32).to(device)
        weights = torch.tensor(weights, dtype=torch.float32).to(device)
        indices = torch.tensor(indices, dtype=torch.int64).to(device)

        # Pre-allocate memory using torch.empty
        best_jig = torch.empty(self.batch_size, dtype=torch.int64)
        best_destination = torch.empty(self.batch_size, dtype=torch.int64)

        with torch.no_grad():
            # Compute next Q-values for all states    
            Q_next_destination, Q_next_jig = policy_dqn(new_states)

            # Compute best destinations
            destination_masks = [destination_mask(domain, new_state_pddl, destination_ids) for new_state_pddl in new_state_pddls]
            best_destination = torch.argmax(Q_next_destination.masked_fill(~torch.stack(destination_masks), LARGE_NEG), dim=1)
            best_destination_obj = destination_ids_tensor[best_destination]

            # Compute best Jigs
            jig_masks = [jig_mask(destination_id, domain, new_state_pddl) for destination_id, new_state_pddl in zip(best_destination_obj, new_state_pddls)]
            best_jig = torch.argmax(Q_next_jig.masked_fill(~torch.stack(jig_masks), LARGE_NEG), dim=1)

            # Use target network to evaluate best actions (Duelling DQN)
            Q_target_destination, Q_target_jig = target_dqn(new_states)
            target_Q_jig = Q_target_jig.gather(1, best_jig.unsqueeze(1)).squeeze(1)
            target_Q_destination = Q_target_destination.gather(1, best_destination.unsqueeze(1)).squeeze(1)

            # Compute Bellman targets
            target_Q_jig = rewards + self.discount_factor * target_Q_jig * (1 - terminated)
            target_Q_destination = rewards + self.discount_factor * target_Q_destination * (1 - terminated)

        # Get current state Q value
        Q_destination, Q_jig = policy_dqn(states)
        Q_jig = Q_jig.gather(1, actions[:, 0].unsqueeze(1)).squeeze(1)
        Q_destination = Q_destination.gather(1, actions[:, 1].unsqueeze(1)).squeeze(1)

       # Compute MSE loss per branch
        td_errors_jig = (target_Q_jig - Q_jig)**2
        td_errors_destination = (target_Q_destination - Q_destination)**2

        # Compute final loss as mean across branches
        jig_loss = torch.mean(td_errors_jig  * weights)
        destination_loss = torch.mean(td_errors_destination  * weights)

        loss = (jig_loss + destination_loss)/2

        # Optimise network
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        # torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 10.0)
        # self.scheduler.step()

        self.loss_history.append(loss.item())

        # TD error for prioritised replay updates
        td_errors = torch.abs(td_errors_jig) + torch.abs(td_errors_destination)
        memory.update_priorities(indices, td_errors)

    def save_graph(self, rewards_per_episode, epsilon_history, loss_history):
        
        fig = plt.figure(1)

        # Plot rewards mean in 100 range
        plt.subplot(1,2,1)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(rewards_per_episode)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-100):x+1])

        plt.plot(mean_rewards)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Reward per Episode without Priority Experience Replay")

        # plt.plot(loss_history)
        # plt.xlabel("Episode")
        # plt.ylabel("Loss")
        # plt.title("Loss per episode")

        # plot epsilon
        plt.subplot(1,2,2)
        plt.plot(epsilon_history)
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.title("Epsilon per episode")

        fig.savefig(self.GRAPH_FILE)

        plt.close(fig)

    def get_reward(self, states_visited, new_state, action_id, terminated, num_jigs):
        '''
        Give reward for the doing actions of unloading beluga, loading beluga, and sending to hangar 
        '''
        if terminated:
            return 5
        elif action_id in [0,1]:
            return 1
        elif action_id in [3]:
            return 2
        else:
            return -10  if new_state in states_visited else -(1/self.maximum_simulation_steps) 
