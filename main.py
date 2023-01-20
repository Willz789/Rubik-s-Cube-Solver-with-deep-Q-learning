from cube import Cube
from training_loop import training_loop, t_load
import pygame, os
import torch
import numpy as np

def update_screen():
    cube.draw_cube(gameDisplay)
    pygame.display.flip()

display_sizes = {"width" : 700, "height" : 500}
colors = {"black" : (0,0,0), "white" : (255,255,255), "red" : (255,0,0), "green" : (0,255,0), "blue" : (0,0,255), "yellow" : (255,255,0), "orange" : (255,140,0)}

training_model = True

if training_model:
    training_loop(display_sizes['height'], colors, learning_rate=0.001, gamma=0.99, eps=1, eps_min=0.02, save_data=True, load_data=False, total_start_scrambles=1)
else:
    pygame.init()
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (250,50)
    gameDisplay = pygame.display.set_mode((display_sizes["width"], display_sizes["height"]))
    gameDisplay.fill(colors["white"])

    pygame.display.set_caption('Rubix')

    n_start_scrambles = 3
    cube = Cube(colors, display_sizes['height'], n_start_scrambles)

    q_net = torch.nn.Sequential(
        torch.nn.Linear(20*24,4096, dtype=torch.float32),
        torch.nn.ReLU(),
        torch.nn.Linear(4096,2048, dtype=torch.float32),
        torch.nn.ReLU(),
        torch.nn.Linear(2048,512, dtype=torch.float32),
        torch.nn.ReLU(),
        torch.nn.Linear(512,12, dtype=torch.float32)
    )
    
    q_net.load_state_dict(torch.load('SavedData/model.pth', map_location=torch.device('cpu')))

    update_screen()

    program_running = True

    while program_running:
        for event in pygame.event.get():
            mods = pygame.key.get_mods()
            if event.type == pygame.QUIT:
                program_running = False
                print("Qutting program")
            # Looks for clicks on keyboard
            if event.type == pygame.KEYDOWN:
                # Quits
                if event.key == pygame.K_q:
                    program_running = False
                    print("Qutting program")
                # Activates the ai
                elif event.key == pygame.K_a and cube.checkDone() == False:
                    print("Calculates best move...")
                    observation = cube.get_onehot_state().flatten()
                    best_move = cube.move_dict[np.argmax(q_net(torch.tensor(observation).float()).detach().numpy())]
                    print(best_move)
                    gameState, reward, done = cube.scramble_cube(best_move)
                    update_screen()
                elif event.key == pygame.K_s:
                    print("Solving cube...")
                    done = cube.checkDone()
                    while done == False:
                        observation = cube.get_onehot_state().flatten()
                        best_move = cube.move_dict[np.argmax(q_net(torch.tensor(observation).float()).detach().numpy())]
                        print(best_move)
                        gameState, reward, done = cube.scramble_cube(best_move)
                        update_screen()
                # Scrambles the cube randomly
                elif event.key == pygame.K_n:
                    cube.reset_cube()
                    cube.start_scramble_cube()
                    print("Scrambling cube")
                    update_screen()
                # Changes the start_scrambles count
                elif event.key == pygame.K_c:
                    changed = False
                    while changed == False:
                        try:
                            new_amount = int(input('Type the new value of n_start_scrambles (type "0" to keep previous value)\nInput: '))
                        except ValueError:
                            print("not a number")
                        else:
                            if new_amount == 0:
                                print(f'n_start_scrambles is still {cube.n_start_scrambles}')
                                break
                            else:
                                print(f'n_start_scrambles is now {new_amount}')
                                cube.n_start_scrambles = new_amount
                                changed = True
                            
                # Resets to completed cube
                elif event.key == pygame.K_o:
                    print("Resetting cube")
                    cube.reset_cube()
                    update_screen()
                # Scrambles the cube once (Hold Left-Shift for inverse scramble)
                elif event.key == pygame.K_u:
                    if mods & pygame.KMOD_LSHIFT or mods & pygame.KMOD_CAPS:
                        gameState, reward, done = cube.scramble_cube("U'")
                    else:
                        gameState, reward, done = cube.scramble_cube("U")
                    update_screen()
                elif event.key == pygame.K_d:
                    if mods & pygame.KMOD_LSHIFT or mods & pygame.KMOD_CAPS:
                        gameState, reward, done = cube.scramble_cube("D'")
                    else:
                        gameState, reward, done = cube.scramble_cube("D")
                    update_screen()
                elif event.key == pygame.K_l:
                    if mods & pygame.KMOD_LSHIFT or mods & pygame.KMOD_CAPS:
                        gameState, reward, done = cube.scramble_cube("L'")
                    else:
                        gameState, reward, done = cube.scramble_cube("L")
                    update_screen()
                elif event.key == pygame.K_r:
                    if mods & pygame.KMOD_LSHIFT or mods & pygame.KMOD_CAPS:
                        gameState, reward, done = cube.scramble_cube("R'")	
                    else:
                        gameState, reward, done = cube.scramble_cube("R")
                        update_screen()
                elif event.key == pygame.K_f:
                    if mods & pygame.KMOD_LSHIFT or mods & pygame.KMOD_CAPS:
                        gameState, reward, done = cube.scramble_cube("F'")
                    else:
                        gameState, reward, done = cube.scramble_cube("F")
                    update_screen()
                elif event.key == pygame.K_b:
                    if mods & pygame.KMOD_LSHIFT or mods & pygame.KMOD_CAPS:
                        gameState, reward, done = cube.scramble_cube("B'")
                    else:
                        gameState, reward, done = cube.scramble_cube("B")
                    update_screen()