import torch

def jig_mask(action_id, domain, state):

    """
    Generates a mask tensor indicating which jigs have applicable actions in the given state.

    Args:
        domain: The domain containing the task and its objects.
        state: The current state of the problem.

    Returns:
        A torch tensor of size equal to the number of jigs, where each element is 1 if the 
        corresponding jig has applicable actions, and 0 otherwise.
    """

    jigs_ids = torch.tensor([domain.task.objects.index(obj) for obj in domain.task.objects if obj.startswith("jig")], dtype=torch.int64)

    # Represents each of the jigs.
    mask = torch.zeros(len(jigs_ids), dtype=torch.bool)
    
    # Get applicable actions
    applicable_actions = domain.get_applicable_actions(state)._elements

    # Extract relevant jig_ids in a tensor for vectorized processing
    jig_ids = torch.tensor([action.args[0] for action in applicable_actions if action.action_id == action_id], dtype=torch.int64)

    # Get valid indices in jigs_ids and update mask
    valid_indices = torch.nonzero(torch.isin(jigs_ids, jig_ids), as_tuple=True)[0]
    mask[valid_indices] = 1
    
    return mask



def action_mask(domain, state):     # maybe add list of used racks? for put down and stack

    # represents each of the 9 action id, e.g. load-beluga, deliver-to-hangar, etc.
    """
    Generates a mask tensor indicating which actions are applicable for a given jig in a given state.

    Args:
        jig_id: The id of the jig.
        domain: The domain containing the task and its objects.
        state: The current state of the problem.

    Returns:
        A torch tensor of size 9, where each element is 1 if the corresponding action is applicable, and 0 otherwise.

    Notes:
        The general action availability depends on the position of the jig in the state. The different cases are detailed below.

        1. Jig in Beluga: 
            - unload-beluga 

        2. Jig in beluga trailer:
            - load-beluga
            - put-down-rack
            - stack-rack

        3. Jig in rack:
            - pick-up-rack
            - unstack-rack

        4. Jig in factory trailer:
            - deliver-to-hangar
            - put-down-rack
            - stack-rack

        5. Jig in hangar:
            - get-from-hangar
    """

    mask = torch.zeros(8, dtype=torch.bool)  

    # Get applicable actions
    applicable_actions = domain.get_applicable_actions(state)._elements

    # Convert applicable actions to a tensor format for vectorized processing
    action_ids = torch.tensor([action.action_id for action in applicable_actions if action.action_id != 8], dtype=torch.int64)

    mask[action_ids] = 1

    return mask
    

def destination_mask(jig_id, action_id, domain, state):
    
    """
    Generates a mask tensor indicating which destinations are applicable for a given jig and action in a given state.

    Args:
        jig_id: The id of the jig.
        action_id: The id of the action.
        domain: The domain containing the task and its objects.
        state: The current state of the problem.

    Returns:
        A torch tensor of size equal to the number of destinations, where each element is 1 if the 
        corresponding destination is applicable, and 0 otherwise.

    Notes:
        The destination availability is determined based on the action type. Each action has specific 
        destinations it can be applied to, which are indicated by the indices in the destination_ids list.

        1. destination index = 1 for action: pick-up-rack .                                                 Index of [6]
        2. destination index = 2 for action: unload-beluga; get-from-hangar; put-down-rack; unstack-rack .  Index of [1,2,4,7]
        3. destination index = 3 for action: load_beluga; deliver-to-hangar; stack-rack.                    Index of [0,3,5]
    """

    destination_ids = get_valid_destination(domain)

    # represents each of the destination.
    mask = torch.zeros(len(destination_ids), dtype=torch.bool)

    # Get applicable actions
    applicable_actions = domain.get_applicable_actions(state)

    # dict for destination id depending on action_id
    destination_indexes = {
        6: 1,  
        1: 2, 2: 2, 4: 2, 7: 2,  
        0: 3, 5: 3,
        3: 4 
    }

    action_id = action_id.item() if type(action_id) == torch.Tensor else action_id

    # Extract jig id and action id from applicable actions
    jig_ids = torch.tensor([action.args[0] for action in applicable_actions._elements])
    action_ids = torch.tensor([action.action_id for action in applicable_actions._elements])

    # Find indices where jig id and action id match
    match_indices = (jig_ids == jig_id) & (action_ids == action_id)

    # Get destination index for matching actions
    destination_index = destination_indexes[action_id]
    destination_objs = torch.tensor([action.args[destination_index] if len(action.args) > destination_index else 0 for action in applicable_actions._elements])

    # Set corresponding indices to 1
    mask[torch.tensor([destination_ids.index(obj) for obj in destination_objs[match_indices]], dtype=torch.int64)] = 1

    # # Iterate over applicable actions and set corresponding indices to 1
    # for action in applicable_actions._elements:
    #     cur_jig_id = action.args[0]  # Extract jig id from action args
    #     if cur_jig_id == jig_id and action.action_id == action_id:
    #         destination_index = destination_indexes[action_id]
    #         destination_obj = action.args[destination_index]
    #         mask[destination_ids.index(destination_obj)] = 1

    return mask
            
    
def generate_applicable_actions(jig, action, destination, domain, state):
        '''
        Details on state value
        1 = Beluga trailer, factory trailer, and hangar
        2 = Racks and their empty space. The second element represent from domain.task in the form of "nxx" where xx define the number of spaces left.
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
        
        if action in [6]:  # case for complete beluga which destination is don't care
            destination_index = 1
        elif action in [1,2,4,7]:
            destination_index = 2
        elif action in [0,5]:
            destination_index = 3
        elif action in [3]:
            destination_index = 4
        else:
            raise ValueError('Invalid action id')

        jig_index = 0

        applicable_actions = domain.get_applicable_actions(state)._elements
        for a in applicable_actions:
            if a.action_id == action and action == 8: # case of beluga-complete
                return a

             # case of other actions which require arguments for jigs and destination
            if destination_index < len(a.args): # avoid out of bounds error when dealing with short argument action (e.g. action_id 2)
                if a.args[jig_index] == jig and a.args[destination_index] == destination:  
                    return a

        return None


def extract_from_actions(action):
        if action.action_id in [6]:  # case for complete beluga which destination is don't care
            destination_index = 1
        elif action.action_id in [1,2,4,7]:
            destination_index = 2
        elif action.action_id in [0,5]:
            destination_index = 3
        elif action.action_id in [3]:
            destination_index = 4
        elif action.action_id in [8]:
            destination_index = None
        else:
            raise ValueError('Invalid action id')

        jig_index = 0
        jig_id = action.args[jig_index]

        # Extract destination_id if applicable
        destination_id = action.args[destination_index] if destination_index is not None else 0

        return jig_id, action.action_id, destination_id

def get_object_name(domain, obj_id):
    objects = domain.task.objects  # List of object names indexed by their IDs
    if 0 <= obj_id < len(objects):
        return objects[obj_id]
    return None  # Return None if out of bounds

def get_valid_destination(domain):
    destination = []
    objects = domain.task.objects
    for index, obj in enumerate(objects):
        if obj.startswith(("beluga_trailer","factory_trailer","rack", "pl", "beluga")):
            destination.append(index)

    return destination


