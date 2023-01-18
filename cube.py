import pygame, os
from tools import rotateMatrix
import numpy as np

class Cube:
    def __init__(self, colors, display_height, n_start_scrambles):
        self.n_start_scrambles = n_start_scrambles
        self.colors = colors

        self.colors_dict = {0 : colors["green"], 1 : colors["white"], 2 : colors["blue"], 3 : colors["yellow"], 4 : colors["red"], 5: colors["orange"]}

        # Size of one piece
        self.width = 50
        self.y_start = display_height/2 - 1.5*self.width
        self.x_start = 50

        self.moves = ["L", "L'", "R", "R'", "F", "F'", "B", "B'", "U", "U'", "D", "D'"]
        self.oppo_move = {  "L" : "L'", "L'" : "L", "R" : "R'", "R'" : "R", "F" : "F'", "F'" : "F", 
                            "B" : "B'", "B'" : "B", "U" : "U'", "U'" : "U", "D" : "D'", "D'" : "D", None : None}
        self.move_dict = {  0 : self.moves[0], 1 : self.moves[1], 2 : self.moves[2], 3 : self.moves[3], 
                            4 : self.moves[4], 5 : self.moves[5], 6 : self.moves[6], 7 : self.moves[7], 
                            8 : self.moves[8], 9 : self.moves[9], 10 : self.moves[10], 11 : self.moves[11]
        }

        # Matrix representation of the cube - faces represented by 3x3 matrices: 
        # 1st: Down-face, 2nd: Left-face, 3rd: Up-face, 4th: Right-face, 5th: Front-face, 6th: Back-face
        self.completed_gameState = [  np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=int),
                            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=int),
                            np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]], dtype=int),
                            np.array([[3, 3, 3], [3, 3, 3], [3, 3, 3]], dtype=int),
                            np.array([[4, 4, 4], [4, 4, 4], [4, 4, 4]], dtype=int),
                            np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=int) ]

        self.gameState = np.copy(self.completed_gameState)
                            
        self.oppo_dict = {0 : 2, 1 : 3, 2 : 0, 3 : 1, 4 : 5, 5 : 4}
        # Priority first sticker
        self.stickers = [   [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,2], [0,2,0], [0,2,1], [0,2,2], #7
                            [1,0,0], [1,0,1], [1,0,2], [1,1,0], [1,1,2], [1,2,0], [1,2,1], [1,2,2], #15
                            [2,0,0], [2,0,1], [2,0,2], [2,1,0], [2,1,2], [2,2,0], [2,2,1], [2,2,2], #23
                            [3,0,0], [3,0,1], [3,0,2], [3,1,0], [3,1,2], [3,2,0], [3,2,1], [3,2,2], #31
                            [4,0,0], [4,0,1], [4,0,2], [4,1,0], [4,1,2], [4,2,0], [4,2,1], [4,2,2], #39
                            [5,0,0], [5,0,1], [5,0,2], [5,1,0], [5,1,2], [5,2,0], [5,2,1], [5,2,2]  #47
                        ]   

        self.prio_stickers = [0,1,2,3,4,5,6,7,9,16,17,18,19,20,21,22,23,30,35,44]

        self.sticker_sets = {   0 : [26, 42], 1 : [41], 2 : [8, 40], 3 : [28], 4 : [11], 5 : [31, 39], 6 : [38], 
                                7 : [13, 37], 9 : [43], 16 : [10, 45], 17 : [46], 18 : [24, 47], 19 : [12], 
                                20 : [27], 21 : [15, 32], 22 : [33], 23 : [29, 34], 30 : [36], 35 : [14], 44 : [25]
                            }
        self.int_encoded_completed_state = np.array([1,3,0,1,0,1,2,0,7,8,11,9,8,9,8,10,9,14,17,23])
        self.encoded_completed_state = np.array([
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        self.faceDict = {"D": 0, "L": 1, "U": 2, "R": 3, "F": 4, "B": 5}
        
    def reset_cube(self):
        self.gameState = np.copy(self.completed_gameState)
        #return self.gameState

    def start_scramble_cube(self):
        done = True
        moves = []
        prev_move = None
        #print("Scrambling cube...")
        for i in range(self.n_start_scrambles):
            move = np.random.choice(self.moves)
            if i != 0:
                while self.oppo_move[prev_move] == move or self.oppo_move[move] == prev_move:
                    move = np.random.choice(self.moves)
            moves.append(move)
            gameState, reward, done = self.scramble_cube(move)
        if done:
            self.start_scramble_cube()
        return moves
    
    def draw_face(self, gameDisplay, x_start, y_start, sticker_colors):
        for i in range(3):
            for j in range(3):
                c = 3*i+j
                pygame.draw.rect(gameDisplay, self.colors_dict[sticker_colors[c]], 
                    pygame.Rect(x_start + j * self.width, y_start + i * self.width, self.width, self.width))
                pygame.draw.rect(gameDisplay, self.colors["black"], 
                    pygame.Rect(x_start + j * self.width, y_start + i * self.width, self.width, self.width), 1)

        pygame.draw.rect(gameDisplay, self.colors["black"], pygame.Rect(self.x_start, self.y_start, self.width*3, self.width*3), 1)

    def draw_cube(self, gameDisplay):
        self.draw_face(gameDisplay, self.x_start, self.y_start, list(self.gameState[0][0]) + list(self.gameState[0][1]) + list(self.gameState[0][2]))
        self.draw_face(gameDisplay, self.x_start + 3 * self.width, self.y_start, list(self.gameState[1][0]) + list(self.gameState[1][1]) + list(self.gameState[1][2]))
        self.draw_face(gameDisplay, self.x_start + 6 * self.width, self.y_start, list(self.gameState[2][0]) + list(self.gameState[2][1]) + list(self.gameState[2][2]))
        self.draw_face(gameDisplay, self.x_start + 9 * self.width, self.y_start, list(self.gameState[3][0]) + list(self.gameState[3][1]) + list(self.gameState[3][2]))
        self.draw_face(gameDisplay, self.x_start + 6 * self.width, self.y_start + 3 * self.width, list(self.gameState[4][0]) + list(self.gameState[4][1]) + list(self.gameState[4][2]))
        self.draw_face(gameDisplay, self.x_start + 6 * self.width, self.y_start - 3 * self.width, list(self.gameState[5][0]) + list(self.gameState[5][1]) + list(self.gameState[5][2]))

    def get_onehot_state(self):
        onehot_state = np.zeros((20,24))
        for i in range(len(self.prio_stickers)):
            sticker_id = self.prio_stickers[i]
            state_found = False
            counter = 0
            s_1 = self.stickers[sticker_id]
            s_2 = self.stickers[self.sticker_sets[sticker_id][0]]
            for j in range(6):
                if state_found:
                    break
                for k in range(6):
                    if state_found:
                        break
                    elif self.oppo_dict[k] != j and k != j:
                        if self.gameState[s_1[0]][s_1[1]][s_1[2]] == j and self.gameState[s_2[0]][s_2[1]][s_2[2]] == k:
                            state_found = True
                            break
                        counter += 1
            onehot_state[i][counter] = 1
        return onehot_state

    def print_current_state(self):
        encoded_state = self.get_onehot_state()
        print(encoded_state, "\n")
        gameState_int = []
        for row in encoded_state:
            gameState_int.append(row.index(1))
        print(gameState_int)

    def checkDone(self):
        encoded_gameState = self.get_onehot_state()
        for i in range(len(encoded_gameState)):
            loc_one = np.where(encoded_gameState[i]==1)[0][0]
            comp_loc_one = self.int_encoded_completed_state[i]
            if loc_one != comp_loc_one:
                return False
        return True

    def scramble_cube(self, action):
        face = self.faceDict[action[0]]

        # Creates lists of squares that move for each possible action
        if face == 0:
            # yellow(2,5,8), red(8,7,6), white(6,3,0), orange(0,1,2)self.
            neighbours = 	[self.gameState[3][0][2], self.gameState[3][1][2], self.gameState[3][2][2], 
                            self.gameState[4][2][2], self.gameState[4][2][1], self.gameState[4][2][0], 
                            self.gameState[1][2][0], self.gameState[1][1][0], self.gameState[1][0][0], 
                            self.gameState[5][0][0], self.gameState[5][0][1], self.gameState[5][0][2]]
            face_line = 	[[3, 0, 2], [3, 1, 2], [3, 2, 2], 
                            [4, 2, 2], [4, 2, 1], [4, 2, 0], 
                            [1, 2, 0], [1, 1, 0], [1, 0, 0], 
                            [5, 0, 0], [5, 0, 1], [5, 0, 2]]

        elif face == 1:
            # blue(6,3,0), orange(6,3,0), green(2,5,8), red(6,3,0)
            neighbours = 	[self.gameState[2][2][0], self.gameState[2][1][0], self.gameState[2][0][0],
                            self.gameState[5][2][0], self.gameState[5][1][0], self.gameState[5][0][0],
                            self.gameState[0][0][2], self.gameState[0][1][2], self.gameState[0][2][2],
                            self.gameState[4][2][0], self.gameState[4][1][0], self.gameState[4][0][0]]
            face_line = 	[[2, 2, 0], [2, 1, 0], [2, 0, 0], 
                            [5, 2, 0], [5, 1, 0], [5, 0, 0], 
                            [0, 0, 2], [0, 1, 2], [0, 2, 2], 
                            [4, 2, 0], [4, 1, 0], [4, 0, 0]]
        elif face == 2:
            # red(0,1,2), yellow(6,3,0), orange(8,7,6), white(2,5,8)
            neighbours = 	[self.gameState[4][0][0], self.gameState[4][0][1], self.gameState[4][0][2],
                            self.gameState[3][2][0], self.gameState[3][1][0], self.gameState[3][0][0],
                            self.gameState[5][2][2], self.gameState[5][2][1], self.gameState[5][2][0],
                            self.gameState[1][0][2], self.gameState[1][1][2], self.gameState[1][2][2]]
            face_line = 	[[4, 0, 0], [4, 0, 1], [4, 0, 2], 
                            [3, 2, 0], [3, 1, 0], [3, 0, 0], 
                            [5, 2, 2], [5, 2, 1], [5, 2, 0], 
                            [1, 0, 2], [1, 1, 2], [1, 2, 2]]
        elif face == 3:
            # blue(2,5,8), red(2,5,8), green(6,3,0), orange(2,5,8)
            neighbours = 	[self.gameState[2][0][2], self.gameState[2][1][2], self.gameState[2][2][2],
                            self.gameState[4][0][2], self.gameState[4][1][2], self.gameState[4][2][2],
                            self.gameState[0][2][0], self.gameState[0][1][0], self.gameState[0][0][0],
                            self.gameState[5][0][2], self.gameState[5][1][2], self.gameState[5][2][2]]
            face_line = 	[[2,0,2], [2,1,2], [2,2,2], 
                            [4,0,2], [4,1,2], [4,2,2], 
                            [0,2,0], [0,1,0], [0,0,0], 
                            [5,0,2], [5,1,2], [5,2,2]]
        elif face == 4:
            # blue(8,7,6), white(8,7,6), green(8,7,6), yellow(8,7,6)
            neighbours = 	[self.gameState[2][2][2], self.gameState[2][2][1], self.gameState[2][2][0],
                            self.gameState[1][2][2], self.gameState[1][2][1], self.gameState[1][2][0],
                            self.gameState[0][2][2], self.gameState[0][2][1], self.gameState[0][2][0],
                            self.gameState[3][2][2], self.gameState[3][2][1], self.gameState[3][2][0]]
            face_line = 	[[2,2,2], [2,2,1], [2,2,0], 
                            [1,2,2], [1,2,1], [1,2,0], 
                            [0,2,2], [0,2,1], [0,2,0], 
                            [3,2,2], [3,2,1], [3,2,0]]
        elif face == 5:
            # blue(0,1,2), yellow(0,1,2), green(0,1,2), white(0,1,2)
            neighbours = 	[self.gameState[2][0][0], self.gameState[2][0][1], self.gameState[2][0][2],
                            self.gameState[3][0][0], self.gameState[3][0][1], self.gameState[3][0][2],
                            self.gameState[0][0][0], self.gameState[0][0][1], self.gameState[0][0][2],
                            self.gameState[1][0][0], self.gameState[1][0][1], self.gameState[1][0][2]]
            face_line = 	[[2,0,0], [2,0,1], [2,0,2], 
                            [3,0,0], [3,0,1], [3,0,2], 
                            [0,0,0], [0,0,1], [0,0,2], 
                            [1,0,0], [1,0,1], [1,0,2]]

        # Uses numpy.roll() to move the squares that should move
        if len(action) == 1:
            neighbours = np.roll(neighbours, 3 if face == 0 else -3)
            self.gameState[face] = rotateMatrix(self.gameState[face], 3 if face == 0 else 1)
        else:
            neighbours = np.roll(neighbours, -3 if face == 0 else 3)
            self.gameState[face] = rotateMatrix(self.gameState[face], 1 if face == 0 else 3)
        # Applies the changes to the cube
        for i in range(len(face_line)):
            self.gameState[face_line[i][0]][face_line[i][1]][face_line[i][2]] = neighbours[i]
        done = self.checkDone()

        return self.get_onehot_state(), 1 if done else -1, done
    