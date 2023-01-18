import torch

def oneHot_int(c_oneHot):
	return c_oneHot.index(1)

def int_oneHot(c_int, size):
    onehot_list = []
    for i in range(size):
        list.append(0)
    onehot_list[c_int] = 1
    return onehot_list

# Rotates a face of the cube
def rotateMatrix(matrix, rotations):
    for i in range(rotations):
        tuples = list(zip(*matrix[::-1]))
        matrix = list([list(elem) for elem in tuples])
    return matrix

def save_network(q_net):
    prev_state_dict = torch.load('SavedData/new_model.pth')
    torch.save(prev_state_dict, 'SavedData/backup_model.pth')
    torch.save(q_net.state_dict, 'SavedData/new_model.pth')

def load_network():
    return torch.load('SavedData/new_model.pth')
