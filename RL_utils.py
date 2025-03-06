import torch

def action_mask(jig_id, domain, state):     # maybe add list of used racks? for put down and stack

    # represents each of the 9 action id, e.g. load-beluga, deliver-to-hangar, etc.
    mask = torch.zeros(9, dtype=torch.float32)  

    # get jig position
    jig_location = None
    jig_data = state.atoms[3]  # The third atom contains jig-location pairs
    for jig, location in jig_data:
        if jig == jig_id:
            jig_location = get_object_name(domain, location)
            break
    '''
    Different cases for action availability realting to the Jig position. 
    Detailed below are the available actions depending on the Jig position.

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

    '''
    if "beluga" in jig_location:
        mask[1] = 1
    elif "beluga_trailer" in jig_location:
        mask[0] = 1
        mask[4] = 1
        mask[5] = 1
    elif "rack" in jig_location:
        mask[6] = 1
        mask[7] = 1
    elif "factory_trailer" in jig_location:
        mask[3] = 1
        mask[4] = 1
        mask[5] = 1
    elif "hangar" in jig_location:
        mask[2] = 1
    else:
        raise ValueError(f"Unknown jig location: {jig_location}")

    return mask

def destination_mask(action_id, domain, state):
    destination_ids = get_valid_destination(domain)

    # represents each of the destination.
    mask = torch.zeros(len(destination_ids), dtype=torch.float32)

    '''
    Different cases for destination availability relating to the Action taken. 
    Detailed below are the available destination depending on the Actions performed.

    1. load-beluga(0): 
        - beluga
    2. unload-beluga(1):
        - beluga_trailer
    3. get-from-hangar(2):
        - factory_trailer
    4. deliver-to-hangar(3):
        - hangar
    5. put-down-rack(4):
        - rack
    6. stack-rack(5):
        - rack
    7. pick-up-rack(6):
        - beluga_trailer
        - factory_trailer
    8. unstack-rack(7):
        - beluga_trailer
        - factory_trailer
    9. beluga-complete(8):
        -N/A
    '''
    # Valid destination prefixes for each action
    action_destinations = {
        0: ["beluga"],  # load-beluga
        1: ["beluga_trailer"],  # unload-beluga
        2: ["factory_trailer"],  # get-from-hangar
        3: ["hangar"],  # deliver-to-hangar
        4: ["rack"],  # put-down-rack
        5: ["rack"],  # stack-rack
        6: ["beluga_trailer", "factory_trailer"],  # pick-up-rack
        7: ["beluga_trailer", "factory_trailer"],  # unstack-rack
        8: []  # beluga-complete (N/A destination)
    }

    valid_prefixes = action_destinations.get(action_id, [])

    for idx, dest in enumerate(destination_ids.keys()):
        if any(dest.startswith(prefix) for prefix in valid_prefixes):
            mask[idx] = 1

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
        elif action in [0,3,5]:
            destination_index = 3
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
        elif action.action_id in [0,3,5]:
            destination_index = 3
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
    destination = dict()
    objects = domain.task.objects
    for index, obj in enumerate(objects):
        if obj.startswith(("beluga_trailer","factory_trailer","rack", "hangar", "beluga")):
            destination[obj] = index

    return destination


