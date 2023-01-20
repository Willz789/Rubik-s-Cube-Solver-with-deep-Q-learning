from cube import Cube
import numpy as np
import torch

# Function for saving neural net parameters, epsilon, n_start_scrambles and game_counter
def t_save(q_net, eps, n_start_scrambles, game_counter, load_data):
    # Moves the last saved model to backup, if there is a last saved model
    if load_data:
        prev_model = torch.load('SavedData/model.pth', map_location=torch.device('cpu'))
        try:
            torch.save(prev_model, 'SavedData/backup_model.pth')
        except PermissionError:
            print("\nBackup save failed!")
        else:
            with open('SavedData/params.txt', 'r') as f:
                params = f.readlines()
                for i in range(len(params)):
                    if i != len(params)-1:
                        params[i] = params[i][:-1]
                    if i == 0:
                        params[i] = float(params[i])
                    else:
                        params[i] = int(params[i])
            with open('SavedData/backup_params.txt', 'w') as f:
                param_string = ""
                for i in range(len(params)):
                    param_string += str(params[i])
                    if i != len(params)-1:
                        param_string += "\n"
                f.write(param_string)

    # Saves the new model for the neural network
    try:
        torch.save(q_net.state_dict(), 'SavedData/model.pth')
        print("Saving q_net...")
    except PermissionError:
        print('\nSave failed!')
    else:
        params = [eps, n_start_scrambles, game_counter]
        with open('SavedData/params.txt', 'w') as f:
            param_string = ""
            for i in range(len(params)):
                param_string += str(params[i])
                if i != len(params)-1:
                    param_string += "\n"
            f.write(param_string)

# Function for loading in neural net parameters, epsilon, n_start_scrambles and game_counter
def t_load():
    print('loading model...')
    filepath_start = 'SavedData/' # Change filepath_start to 'SavedData/backup_' for loading backup_model and backup_params otherwise 'SavedData/'
    model = torch.load(filepath_start + 'model.pth', map_location=torch.device('cpu')) 
    with open(filepath_start + 'params.txt', 'r') as f:
            params = f.readlines()
            for i in range(len(params)):
                if i != len(params)-1:
                    params[i] = params[i][:-1]
                if i == 0:
                    params[i] = float(params[i])
                else:
                    params[i] = int(params[i])
    return model, params

# Parameters for each scramble complexity
buffer_size_dict = {1 : 1000, 2 : 2000, 3 : 20000, 4 : 150000}
batch_size_dict = {1 : 16, 2 : 32, 3 : 128, 4 : 512}
eps_dec_dict = {1 : 0.0005, 2 : 0.00025, 3 : 0.00005, 4 : 0.000025}
accuracy_set_sizes_dict = {1 : 50, 2 : 100, 3 : 500, 4 : 750}

# Main training loop function for the neural net
def training_loop(display_height, colors, learning_rate, gamma, eps, eps_min, save_data, load_data, total_start_scrambles):
    
    # Creates the cube
    cube = Cube(colors, display_height, 1)

    torch.set_default_dtype(torch.float32)

    # Creates the DNN
    q_net = torch.nn.Sequential(
        torch.nn.Linear(20*24,4096, dtype=torch.float32),
        torch.nn.ReLU(),
        torch.nn.Linear(4096,2048, dtype=torch.float32),
        torch.nn.ReLU(),
        torch.nn.Linear(2048,512, dtype=torch.float32),
        torch.nn.ReLU(),
        torch.nn.Linear(512,12, dtype=torch.float32)
    )

    completed_loaded_games = 0

    # Loads previously saved model and parameters
    if load_data:
        loaded_q_net, params = t_load()
        q_net.load_state_dict(loaded_q_net)
        eps = params[0]
        cube.n_start_scrambles = params[1]
        completed_loaded_games = params[2]

    # Loads some parameters
    buffer_size = buffer_size_dict[cube.n_start_scrambles]
    batch_size = batch_size_dict[cube.n_start_scrambles]
    eps_dec = eps_dec_dict[cube.n_start_scrambles]

    # Creates optimizer and loss-function
    optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
    loss = torch.nn.MSELoss()

    action_space = np.arange(12)

    # Creates buffers
    obs_buffer = np.zeros((buffer_size, 20*24))
    obs_next_buffer = np.zeros((buffer_size, 20*24))
    action_buffer = np.zeros(buffer_size)
    reward_buffer = np.zeros(buffer_size)
    terminal_buffer = np.zeros(buffer_size)

    for i in range(cube.n_start_scrambles-1,total_start_scrambles,1):
        step_count = 0 # Counts moves made
        j = completed_loaded_games # Counts games played

        # To store sample accuracies
        accuracy_set_size = accuracy_set_sizes_dict[cube.n_start_scrambles]
        accuracy_set = np.array([])
        top_accuracy = 0
        accuracy = 0
        game_counts = []
        accuracies = []

        # Takes actions until it reaches 100% accuracy
        while accuracy < 100 or step_count < buffer_size:
            # Reset game
            cube.reset_cube()
            start_moves = cube.start_scramble_cube()
            observation = cube.get_onehot_state().flatten()
            
            done = False
            move_counter = 0
            moves = [] # To contain list of moves it uses (used for printing results)
            while move_counter < cube.n_start_scrambles:
                move_counter += 1

                # Reduce epsilon greedy action selection
                eps = np.maximum(eps - eps_dec, eps_min)

                # Action selection
                if np.random.rand() < eps and move_counter == 1:
                    random = True # (used for printing results)
                    action = np.random.choice(action_space)
                else:
                    if move_counter == 1:
                        random = False # (used for printing results)
                    action = np.argmax(q_net(torch.tensor(observation).float()).detach().numpy())
                # Step environment
                move = cube.move_dict[action]
                observation_next, reward, done = cube.scramble_cube(move)

                moves.append(move) # (used for printing results)

                observation_next = observation_next.flatten()

                # Store data in buffer
                buf_idx = step_count % buffer_size
                obs_buffer[buf_idx] = observation
                obs_next_buffer[buf_idx] = observation_next
                reward_buffer[buf_idx] = reward
                action_buffer[buf_idx] = action
                terminal_buffer[buf_idx] = done

                # Update observation
                observation = observation_next

                # Update neural network
                if step_count > buffer_size:
                    # Choose minibatch
                    batch_idx = np.random.choice(buffer_size, size=batch_size)

                    # Compute loss
                    out = q_net(torch.tensor(obs_buffer[batch_idx]).float())
                    q_val = out[np.arange(batch_size), action_buffer[batch_idx]]
                    out_next = q_net(torch.tensor(obs_next_buffer[batch_idx]).float())
                    with torch.no_grad():
                        target = torch.tensor(reward_buffer[batch_idx]).float() + \
                            gamma * torch.max(out_next, dim=1).values * (1 - terminal_buffer[batch_idx]) # Remove "1-"terminal_buffer[batch_idx] ?
                    l = loss(q_val, target.float())

                    # Take gradient step
                    optimizer.zero_grad()
                    l.backward()
                    optimizer.step()
                
                step_count += 1
                if done: break
            
            # adds new game to accuracy_set if it wasn't random
            if not random:
              if done:
                accuracy_set = np.append(accuracy_set, 1)
              else:
                accuracy_set = np.append(accuracy_set, 0)
            # Computes and prints the results from each game and accuracy from each sample
            print(f'done: {done}, start: {start_moves}, moves: {moves}, random: {random}, eps: {eps}, n: {j}')
            j+=1
            if j % accuracy_set_size == 0 and len(accuracy_set) != 0:
                accuracy = int(round(np.mean(accuracy_set)*100))
                accuracy_set = np.array([])
                accuracies.append(accuracies)
                game_counts.append(j)
                if accuracy > top_accuracy:
                    # Saves the current model
                    if save_data:
                        t_save(q_net, eps, cube.n_start_scrambles, j, load_data)
                        load_data = True
                    top_accuracy = accuracy
                print(f'Accuracy: {accuracy}% after {len(accuracies)} samples')

        # Resets epsilon and increases scramble complexity
        if total_start_scrambles > cube.n_start_scrambles:
            eps = 1
            cube.n_start_scrambles += 1
            print(f'\nIncrementing n_start_scrambles from {cube.n_start_scrambles-1} to {cube.n_start_scrambles}\n')