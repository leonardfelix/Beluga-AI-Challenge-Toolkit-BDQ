def action_mask(jig, domain, state):
    pass

def generate_applicable_actions(jig, action, destination, domain, state):
        '''
        Details on state value
        1 = Beluga trailer, factory trailer, and hangar
        2 = Racks
        3 = Jigs and their positions

        7 = jigs and their goal production line (1st jigs that's needed)
        8/9 = return to beluga jig type?
        
        
        Details on action arguments
        See https://github.com/TUPLES-Trustworthy-AI/Beluga-AI-Challenge/blob/main-competition/scalability/README.md

        args[0] is always the jig type


        Set the index of destination depending on the action_id given

        1. destination index = 1 for action: pick-up-rack .                                                 Index of [6]
        2. destination index = 2 for action: unload-beluga; get-from-hangar; put-down-rack; unstack-rack .  Index of [1,2,4,7]
        3. destination index = 3 for action: load_beluga; deliver-to-hangar; stack-rack.                    Index of [0,3,5]

        '''
        
        if action in [6]:
            destination_index = 1
        elif action in [1,2,4,7]:
            destination_index = 2
        elif action in [0,3,5]:
            destination_index = 3
        elif action in [8]:     # case for complete beluga which don't have a destination
            destination_index = None
        else:
            raise ValueError('Invalid action id')

        jig_index = 0

        applicable_actions = domain.get_applicable_actions(state)._elements
        for a in applicable_actions:
            if a.action_id == action and action == 8: # case of beluga-complete
                return a

            if a.args[jig_index] == jig and a.args[destination_index] == destination:   # case of other actions which require arguments for jigs and destination
                return a
            print(a)
            print(a.args)

        return None

