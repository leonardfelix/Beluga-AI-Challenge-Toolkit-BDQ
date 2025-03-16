from evaluation.planner_api import DeterministicPlannerAPI
from Beluga_custom_GYM import CustomBelugaGymCompatibleDomain
from skd_domains.skd_pddl_domain import SkdPDDLDomain
from evaluation.planner_examples import _skd_action_to_beluga_action
from evaluation.planner_api import BelugaPlan
from beluga_lib.beluga_problem import BelugaProblem

from RL_agent import Agent

class BranchingDQNDeterministicPlanner(DeterministicPlannerAPI):

    def __init__(self,
                 max_steps : int = 100,
                 classic : bool = False):
        self.classic = classic
        self.max_steps = max_steps

    def setup(self):
        pass

    def build_plan(self, prb : BelugaProblem):
        # Build a SKD domain
        domain = SkdPDDLDomain(prb, problem_name='server_side_domain',
                               classic=self.classic)

        state = domain.reset()
        max_atoms = max(sum(len(args) for args in atom) for atom in state.atoms)

        print(
            "Creating Gym-compatible domain, i.e. containing array-like spaces for actions and states"
        )
        gym_compatible_domain = CustomBelugaGymCompatibleDomain(
            skd_beluga_domain=domain,
            max_fluent_value=0,
            max_nb_atoms_or_fluents=10,
            max_nb_steps=0,
            max_atom_args = max_atoms*3
        )

        dql = Agent("belugaAI", domain, gym_compatible_domain)
        best_actions, best_reward = dql.run(is_training=True)

        # Translate the plan
        res = BelugaPlan()
        count = 0
        for a in best_actions:
            if count <= self.max_steps:
                ba = _skd_action_to_beluga_action(action=a, domain=domain, classic=self.classic)
                res.append(ba)
                count += 1
            else:
                break
        
        # Cleanup
        domain.cleanup()

        # Return the result
        return res


        # action_space = domain.get_action_space()
        # observation_space = domain.get_observation_space()

        # res = BelugaPlan()
        # s = domain.reset()
        # for step in range(self.max_steps):
        #     # Stop the process if the goal has been reached
        #     if domain._is_terminal(s):
        #         break
        #     # Determine applicable actions
        #     actions = domain.get_applicable_actions(s)
        #     # Stop the process if a dead-end has been reached
        #     if len(actions.get_elements()) == 0:
        #         break
        #     # Sample an applicable action
        #     a = actions.sample()
        #     # Convert the action in the competition format
        #     ba = _skd_action_to_beluga_action(action=a, domain=domain, classic=self.classic)
        #     # Store the action in the plan
        #     res.append(ba)
        #     # Apply the action and move to the next state
        #     o = domain.step(a)
        #     s = o.observation

        # # # Return the plan
        # # return res

        # # Cleanup
        # domain.cleanup()

        # # Return the result
        # return res