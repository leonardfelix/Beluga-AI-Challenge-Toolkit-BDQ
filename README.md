# Beluga-AI-Challenge-Toolkit-BDQ

This repository explores solutions using [**Branching Dueling Q-Networks (BDQ)**](https://arxiv.org/abs/1711.08946)  for **sequential decision making** in the **Airbus BelugaXL cargo logistics** planning problem. The toolkit is developed to tackle the Airbus Beluga AI Challenge and provides an implementation using a custom reinforcement learning agent.

## Overview

The logistics problem involves selecting an optimal sequence of actions which could be broken down to choosing the appropriate **jig**, **action**, and **destination** to efficiently load and transport cargo. This repository models the problem as a **sequential decision process**, where each step (e.g., jig selection, action selection, destination selection) is conditioned on prior decisions (state).

The **Branching Q-Network (BDQ)** is used to decompose the action space and learn Q-values for each action component separately, allowing for efficient exploration and evaluation of multi-branch decision spaces.

---

## Repository Structure

### 🔁 `Main Code`
The core implementation of the **Branching Q-Network (BDQ)** algorithm used for solving the logistics problem.

- `RL_agent.py`: BDQ agent implementation.
- `prioritised_experience_replay.py`: Prioritised experience replay buffer used for training.
- `Beluga_custom_GYM.py`: Custom GYM environment used to train the BDQ model for the Airbus Beluga cargo logistics problem.

### ⚙️ `Hyperparameters`
`hyperparameters.yaml` contains configuration used for training and evaluation. You can modify these to experiment with different learning rates, batch sizes, discount factors, and more.

### 📁 `Example Instances`
Example problem cases for evaluation. Each folder represents a distinct instance of the logistics scheduling problem.

- `_three_jigs/`
- `_four_jigs/`
- `_six_jigs/`

Each directory contains JSON files such as:

- `problem_s3_j3_r2_oc00_f3.json`
- `problem_s3_j4_r2_oc00_f3.json`

which represent logistics problems with different configurations (e.g., number of jigs, racks, etc.).

---

## Getting Started
1. **Clone the repository**:
```bash
  git clone https://github.com/leonardfelix/Beluga-AI-Challenge-Toolkit-BDQ.git
  cd Beluga-AI-Challenge-Toolkit-BDQ
```
2. **Install required packages**:
  ```bash
    pip install -r requirements.txt
  ```
3. **Train and evaluate the BDQ agent**:
   
  Modify the `hyperparameters.yaml` file if needed and run:
  ```bash
  python evaluate_instance.py --input "[folder_name]/[problem_name.json]"
  ```
  e.g.
  ```bash
  python evaluate_instance.py --input "_three_jigs/problem_s3_j3_r2_oc00_f3.json"
  ```
  This will:
  - Load the specified logistics problem.
  - Train the BDQ agent on the environment to make sequential decisions.
  - Output the resulting plan and statistics.


## Notes
- The BDQ architecture is particularly well-suited for hierarchical or compositional action spaces.
- You can add your own problem instances using the JSON schema of the existing files or generate using '`generate_instance.py`'.

## Acknowledgements
This repository was developed as part of the Airbus Beluga AI Challenge and is based on the logistics planning task involving BelugaXL cargo operations. This code was built upon the [Beluga-AI-Challenge-Toolkit](https://github.com/TUPLES-Trustworthy-AI/Beluga-AI-Challenge-Toolkit.git), developed by the team at Airbus and Tuples Trustworthy AI.
