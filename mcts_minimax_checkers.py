import copy
import sys
import numpy as np
import math
import random
import pickle #tried using this instead of deepCopy()

NUM_COLS = 8 #board size
WHITE = 1
NOBODY = 0
BLACK = -1

TIE = 2

WHITE_TO_PLAY = True #set turn

new_possible = [] #global array for when we need to append moves froms jumps function to the rest of the moves in getMoves()

class Piece():
    def __init__(self, color, x, y):
        self.player = color #either WHITE or BLACK
        self.position = (x,y) #location in numpy array
        self.queen = False

    def make_queen(self):
        self.queen = True


class Board():
    def __init__(self):
        self.board = np.zeros((NUM_COLS, NUM_COLS))
        self.parent = None #initial state
        self.possible_moves = None #must call getMoves()
        self.wins = 0 
        self.playouts = 0
        self.turn = WHITE_TO_PLAY #the turn for the current board
        self.children = []
        self.white_pieces = []
        self.black_pieces = []
        self.game_over = False 
        self.board_score = 0
        #creates board state in numpy array and creates list of black and white piece objects with their stored locations
        for x in range(12):
            space = 41 + (2 * x)
            row = space // 8
            col = (space % 8)
            if row == 5 or row == 7:
                col -= 1
            new_piece = Piece(WHITE, row, col)
            self.board[row][col] = WHITE
            self.white_pieces.append(new_piece)
        for x in range(12):
            space = 1 + (2 * x)
            row = space // 8
            col = (space % 8)
            if row == 1:
                col -= 1
            new_piece = Piece(BLACK, row, col)
            self.board[row][col] = BLACK
            self.black_pieces.append(new_piece)
    
    #function to determine if the game is over (one player lost all pieces)
    def endGame(self):
        if len(self.black_pieces) == 0:
            self.game_over = True
            self.board_score = math.inf
        elif len(self.white_pieces) == 0:
            self.game_over = True
            self.board_score = -math.inf

    #scoring function primarily used in minimax but also in our MCTS to speed up choosing move
    #heuristic is simply the number of white pieces - the number of black pieces with double "points" given if the piece is a queen
    def score(self):
        self.board_score = 0
        for x in self.white_pieces:
            if x.queen: 
                self.board_score += 2
            else:
                self.board_score += 1
        for x in self.black_pieces:
            if x.queen:
                self.board_score -= 2
            else:
                self.board_score -= 1

    #function to get all possible moves
    def getMoves(self):
        global new_possible
        possible = [] #to append Board states to
        if self.turn == WHITE_TO_PLAY: #white moves
            for x in self.white_pieces: #must get all moves for each piece
                if x.queen == False: #piece can move in four directions if queen and only 2 otherwise
                    moves = [(-1, 1), (-1, -1)]
                else:
                    moves = [(1, 1), (1, -1), (-1, -1), (-1,1)]
                for y in moves: #for all moves 'y' from each piece 'x'
                    #checks to see if the move is in bounds of the board
                    if (x.position[0] + y[0] >= 0) and (x.position[0] + y[0] < NUM_COLS) and (x.position[1] + y[1] >= 0) and (x.position[1] + y[1] < NUM_COLS):
                        #checks to see if there are no pieces where 'x' is trying to move to
                        if (self.board[x.position[0] + y[0]][x.position[1] + y[1]] == NOBODY):
                            #this is a legal move so we append it to our moves as a deepCopy as to not edit our actual state
                            toAppend = copy.deepcopy(self)
                            for z in toAppend.white_pieces:
                                #we need to next move the piece but cannot move the piece on our board for 'x', but instead for our 'toAppend' board
                                #once we have located the same piece on our 'toAppend' board we can make the move
                                if z.position == x.position:
                                    toAppend.move(z, z.position[0] + y[0], z.position[1] + y[1])
                                    #helper function to make the move
                                    toAppend.turn = not self.turn
                                    #set the turn to the next player
                                    toAppend.parent = self
                                    toAppend.wins = 0
                                    toAppend.playouts = 0
                                    #above needed for MCTS
                                    toAppend.possible_moves = None
                                    #explained in paper on why this variable was created
                                    possible.append(toAppend)  
                        elif (self.board[x.position[0] + y[0]][x.position[1] + y[1]] == BLACK):
                            #if the space the white piece was trying to move was inhabbited by a black piece check to see if we can take the piece
                            self.jumps(x, WHITE)
                            for q in new_possible:
                                q.turn = not self.turn
                                q.parent = self
                                q.wins = 0
                                q.playouts = 0
                                q.possible_moves = None
                                possible.append(q)
                            new_possible = []
                            #global array for storing moves and retrieving from helper function set back to empty
        else: #Black's move
            for x in self.black_pieces:
                if x.queen == False:
                    moves = [(1, 1), (1, -1)]
                else:
                    moves = [(1, 1), (1, -1), (-1, -1), (-1,1)]
                for y in moves: 
                    if (x.position[0] + y[0] >= 0) and (x.position[0] + y[0] < NUM_COLS) and (x.position[1] + y[1] >= 0) and (x.position[1] + y[1] < NUM_COLS):
                        if (self.board[x.position[0] + y[0]][x.position[1] + y[1]] == NOBODY):
                            toAppend = copy.deepcopy(self)
                            for z in toAppend.black_pieces:
                                if z.position == x.position:
                                    toAppend.move(z, z.position[0] + y[0], z.position[1] + y[1])
                                    toAppend.turn = not self.turn
                                    toAppend.parent = self
                                    toAppend.wins = 0
                                    toAppend.playouts = 0
                                    toAppend.possible_moves = None
                                    possible.append(toAppend) 
                        elif (self.board[x.position[0] + y[0]][x.position[1] + y[1]] == WHITE):
                            self.jumps(x, BLACK)
                            for q in new_possible:
                                q.turn = not self.turn
                                q.parent = self
                                q.wins = 0
                                q.playouts = 0
                                q.possible_moves = None
                                possible.append(q)
                            new_possible = []
        return possible
        
    def jumps(self, x, color):
        global new_possible
        if color == WHITE: #checks for moves where piece jumps over a piece of opposite color
            if x.queen == False: #checks to see how many directions piece can move in depending if it is a queen or not
                new_moves = [(-2, 2), (-2, -2)]
            else:
                new_moves = [(2, 2), (2, -2), (-2, -2), (-2,2)]
            for z in new_moves: 
                if (x.position[0] + z[0] >= 0) and (x.position[0] + z[0] < NUM_COLS) and (x.position[1] + z[1] >= 0) and (x.position[1] + z[1] < NUM_COLS):
                    #checks if move is in bounds
                    if (self.board[int(x.position[0] + (z[0]/2))][int(x.position[1] + (z[1]/2))] == BLACK) and (self.board[int(x.position[0] + (z[0]))][int(x.position[1] + (z[1]))] == NOBODY):
                        #checks if the spot it is trying to move to is uninhabited on our board and that the piece it is jumping over is of opposite color
                        toAppend = copy.deepcopy(self)
                        for y in toAppend.white_pieces:
                            #finds piece to move in the new board state
                            if y.position == x.position:
                                toAppend.move(y, y.position[0] + z[0], y.position[1] + z[1])
                                toAppend.turn = not self.turn
                                break
                        for bp in toAppend.black_pieces:
                            #finds piece to delete in new board state using helper function
                            if bp.position == (int(x.position[0] + (z[0]/2)), int(x.position[1] + (z[1]/2))):
                                toAppend.delete(bp)
                        #this move is legal so it must be appended
                        new_possible.append(toAppend)
                        #we must check if the piece can jump over multiple pieces of opposite color on one turn
                        #we do so by creating a copy of the board and calling our jumps function again
                        toAppend2 = copy.deepcopy(toAppend)
                        for q in toAppend2.white_pieces:
                            if y.position == q.position:
                                toAppend2.turn = self.turn
                                toAppend2.jumps(y, WHITE)
        if color == BLACK: #same logic as above
            if x.queen == False:
                new_moves = [(2, 2), (2, -2)]
            else:
                new_moves = [(2, 2), (2, -2), (-2, -2), (-2,2)]
            for z in new_moves: 
                if (x.position[0] + z[0] >= 0) and (x.position[0] + z[0] < NUM_COLS) and (x.position[1] + z[1] >= 0) and (x.position[1] + z[1] < NUM_COLS):
                    if (self.board[int(x.position[0] + (z[0]/2))][int(x.position[1] + (z[1]/2))] == WHITE) and (self.board[int(x.position[0] + (z[0]))][int(x.position[1] + (z[1]))] == NOBODY):
                        toAppend = copy.deepcopy(self)
                        for y in toAppend.black_pieces:
                            if y.position == x.position:
                                toAppend.move(y, y.position[0] + z[0], y.position[1] + z[1])
                                toAppend.turn = not self.turn
                                break
                        for wp in toAppend.white_pieces:
                            if wp.position == (int(x.position[0] + (z[0]/2)), int(x.position[1] + (z[1]/2))):
                                toAppend.delete(wp)   
                        new_possible.append(toAppend)
                        toAppend2 = copy.deepcopy(toAppend)
                        for q in toAppend2.black_pieces:
                            if y.position == q.position:
                                toAppend2.turn = self.turn
                                toAppend2.jumps(y, BLACK)
    
    #helper function to move piece from current location to new 'x' and 'y'
    def move(self, toMove, x, y):
        #checks to see if the move was to take another piece and if so, deletes the piece from the board
        if (abs(toMove.position[0] - x) == 2) and (abs(toMove.position[1] - y) == 2):
            if toMove.player == WHITE:
                for bp in self.black_pieces:
                        if bp.position == (int((toMove.position[0] + (x))/2), int((toMove.position[1] + (y))/2)):
                            self.delete(bp)  
            else:
                for wp in self.white_pieces:
                        if wp.position == (int((toMove.position[0] + (x))/2), int((toMove.position[1] + (y))/2)):
                            self.delete(wp)  
        (a, b) = toMove.position
        self.board[a][b] = NOBODY
        #changes the numpy array
        toMove.position = (x,y)
        #changes the piece object
        self.board[x][y] = toMove.player
        if (toMove.player == WHITE and x == 0) or (toMove.player == BLACK and x == 7):
            #checks to see if the move made the piece a queen
            toMove.make_queen()

    #helper function to delete a piece
    def delete(self, toDelete):
        #deletes piece from numpy array then removes it from respective pieces array
        self.board[toDelete.position[0]][toDelete.position[1]] = NOBODY
        if toDelete.player == WHITE:
            self.white_pieces.remove(toDelete)
        else:
            self.black_pieces.remove(toDelete)

#UCB1 calculator for MCTS
def UCB1(node):
        deterministic_part = node.wins/node.playouts
        explore_part = math.sqrt(2 * math.log(node.parent.playouts)/node.playouts)
        return deterministic_part + explore_part

#once all children have been explored, selection calls this function to find the best calculated move so far to simulate for
def UCT(state):
    bestval = float('-inf')
    bestnode = None
    for node in state.children:
        #the UCB1 value for each board state is calculated
        val = UCB1(node)
        if val > bestval:
          bestval = val
          bestnode = node
    return bestnode

def selection(board):
    current_node = board
    #the following if else is meant to allow us to call getMoves() just once from this board state, store the result in the possible_moves variable of the board object
    #then child board states are appended from this class variables instead of having to call getMoves() over and over
    #this is repeated multiple times throughout the code
    if current_node.possible_moves == None:
      possible_children = current_node.getMoves()
      current_node.possible_moves = possible_children
    else:
      possible_children = current_node.possible_moves
    #once all child nodes have been explored at least once
    while(len(current_node.children) == len(possible_children)):
        #choose best child currently to simulate for
        best_child = UCT(current_node)
        current_node = best_child
        if current_node.possible_moves == None:
          possible_children = current_node.getMoves()
          current_node.possible_moves = possible_children
        else:
          possible_children = current_node.possible_moves
        #once we get to a node that has not been explored we choose this node to expand from
        if (len(possible_children) == 0):
            return current_node, possible_children
    return current_node, possible_children


def expansion(parent, possible_children):
    #if we have made it to a leaf node with no children then we have our board for simulation
    if len(possible_children) == 0:
        return parent
    #find move that has not been explored yet
    for move in possible_children:
        move_found = False
        #check if move has already been explored
        for child in parent.children:
            #how to compared numpy boards
            if (move.board == child.board).all():
                move_found = True
                break
        #checks if we found the node for simulation that has not been simulated for
        if not move_found:
            if move.possible_moves == None:
              moves = move.getMoves()
              move.possible_moves = moves
            else:
              moves = move.possible_moves
            #if there are no legal moves for this player (unlikely in checkers)
            if len(moves) == 0:
                turn = parent.turn
            else:
                turn = not parent.turn
            move.turn = turn
            parent.children.append(move)
            return move
    return None


def simulation(node):
    #call helper functions to update the board score and if the game is over
    node.endGame()
    node.score()
    move = 0
    while not node.game_over:
        #while players can keep moving make up to 25 moves then score the board since calling getMoves() over and over takes far too long
        if move >= 25:
          node.score()
          node.endGame()
          break
        node.score()
        legal_moves = node.getMoves()
        if legal_moves:
            node = random.choice(legal_moves)
        node.endGame()
        node.score()
        move += 1
        #update the score and if the game is over each move
    #declare "winner" of this simulate by whichever player had a higher score in our heuristic function
    if node.board_score > 0:
        return True
    else:
        return False


def backpropagation(node, winner):
    #update the playouts and who was the winner of each simulation
    while node is not None:
        node.playouts += 1
        if node.turn and not winner:
            node.wins += 1
        if not node.turn and winner:
            node.wins +=1
        node = node.parent

def MCTS_choice(board, iterations):
    #function to actually call our MCTS steps a specified number of times as 'iterations' to simulate for the best move from one given board state
    for x in range(iterations):
        current_node, possible_children = selection(board)
        new_node = expansion(current_node, possible_children)
        white_win = simulation(new_node)
        backpropagation(new_node, white_win)
    #whichever node was explored most is the one chosen as the best move
    max_playouts = 0
    best_child = None
    for child in board.children:
        if child.playouts > max_playouts:
            max_playouts = child.playouts
            best_child = child
    return best_child

#call function to see MCTS play against itself
def play_MCTS():
    board = Board()
    while board.game_over == False:
        moves = board.getMoves()
        print()
        if moves:
            board = MCTS_choice(board, 10)
            print("WHITE MOVE:")
            print(board.board)
        else:
            print("No moves this turn")
            board.turn = not board.turn
        print()
        moves = board.getMoves()
        if moves:
            board = MCTS_choice(board, 10)
            print("BLACK MOVE: ")
            print(board.board)
        else:
            print("No moves this turn")
            board.turn = not board.turn
        board.endGame()
    board.score()
    if board.board_score > 0:
        print("White wins")
    elif board.board_score < 0:
        print("Black wins")
    else:
        print("tie")

# play()

#call function to play against MCTS
def play_against_MCTS():
    board = Board()
    while board.game_over == False:
        moves = board.getMoves()
        print()
        if moves:
            board = MCTS_choice(board, 10)
            print("WHITE MOVE:")
            print(board.board)
        else:
            print("No moves this turn")
            board.turn = not board.turn
        print()
        if (len(board.getMoves()) != 0):
                print("Specify the black piece you would like to move by index 0-7:")
                x = input("row:")
                y = input("col:")
                print("Where would you like to move? (index 0-7)")
                move_x = input("row:")
                move_y = input("col:")
                print(board.board[int(x)][int(y)])
                print(board.board[int(move_x)][int(move_y)])
                for bp in board.black_pieces:
                    if bp.position == (int(x), int(y)):
                        board.move(bp, int(move_x), int(move_y)) 
                board.turn = WHITE_TO_PLAY
                print("YOUR/BLACK's MOVE: ")
                print(board.board)
        board.endGame()
    board.score()
    if board.board_score > 0:
        print("White wins")
    elif board.board_score < 0:
        print("Black wins")
    else:
        print("tie")

#MAX function for minimax with alpha-beta pruning
def maximize (board, depth, alpha, beta):
    #update the score and if the game is over each turn
    board.endGame()
    board.score()
    if board.game_over == True and board.board_score > 0:
        return math.inf, board
    elif board.game_over == True and board.board_score < 0:
        return -math.inf, board
    #if we reached our depth we set for minimax then start to score board states
    elif depth == 0:
        board.score()
        return board.board_score, board
    #if we have not reached ou depth make moves and score moves from each possible move the opposite player could make
    max_state = None
    max_score = -math.inf
    board.turn = WHITE_TO_PLAY
    for state in board.getMoves():
        #recursive call
        iter_score, iteration = minimize(state, depth - 1, alpha, beta)
        if iter_score > max_score:
            max_score = iter_score
            max_state = state
        #alpha-beta pruning
        alpha = max(alpha, iter_score)
        if beta <= alpha:
            break
    return max_score, max_state

#MIN player
def minimize (board, depth, alpha, beta):
    board.endGame()
    board.score()
    if board.game_over == True and board.board_score < 0:
        return -math.inf, board
    elif board.game_over == True and board.board_score > 0:
        return math.inf, board
    elif depth == 0:
        board.score()
        return board.board_score, board
    min_state = None
    min_score = math.inf
    board.turn = not WHITE_TO_PLAY
    for state in board.getMoves():
        iter_score, iteration = maximize(state, depth - 1, alpha, beta)
        if iter_score < min_score:
            min_score = iter_score
            min_state = state
        beta = min(beta, iter_score)
        if beta <= alpha:
            break
    return min_score, min_state

#call function to play against minimax
def play_against_minimax(board):
    print(board.board)
    while board.game_over != True:
        if board.turn == WHITE_TO_PLAY:
            if (len(board.getMoves()) != 0):
                board = maximize(board, 3, -math.inf, math.inf)[1]
                board.turn = not WHITE_TO_PLAY
                print("AI/WHITE's MOVE: ")
                print(board.board)
        else:
            if (len(board.getMoves()) != 0):
                print("Specify the black piece you would like to move by index 0-7:")
                x = input("row:")
                y = input("col:")
                print("Where would you like to move? (index 0-7) ")
                move_x = input("row:")
                move_y = input("col:")
                print(board.board[int(x)][int(y)])
                print(board.board[int(move_x)][int(move_y)])
                for bp in board.black_pieces:
                    if bp.position == (int(x), int(y)):
                        board.move(bp, int(move_x), int(move_y)) 
                board.turn = WHITE_TO_PLAY
                print("YOUR/BLACK's MOVE: ")
                print(board.board)
        board.endGame()
        board.score()
    if board.board_score > 0:
        print("White won")
    else:
        print("Black won")

#call function to see minimax play against itself
def play_minimax(board):
    print(board.board)
    while board.game_over != True:
        if board.turn == WHITE_TO_PLAY:
            if (len(board.getMoves()) != 0):
                board = maximize(board, 3, -math.inf, math.inf)[1]
                board.turn = not WHITE_TO_PLAY
                print("AI/WHITE's MOVE: ")
                print(board.board)
        else:
            if (len(board.getMoves()) != 0):
                board = minimize(board, 3, -math.inf, math.inf)[1]
                board.turn = WHITE_TO_PLAY
                print("AI/BLACK's MOVE: ")
                print(board.board)
        board.endGame()
        board.score()
    if board.board_score > 0:
        print("White won")
    else:
        print("Black won")

#AI Minimax (White) plays MCTS (Black)
def MCTS_play_minimax(board):
    print(board.board)
    while board.game_over != True:
        if board.turn == WHITE_TO_PLAY:
            if (len(board.getMoves()) != 0):
                board = maximize(board, 3, -math.inf, math.inf)[1]
                board.turn = not WHITE_TO_PLAY
                print("AI/WHITE's MOVE: ")
                print(board.board)
            else:
                print("No moves this turn")
                board.turn = not board.turn
        
            if (len(board.getMoves()) != 0):
                board = MCTS_choice(board, 10)
                board.turn = WHITE_TO_PLAY
                print("AI/BLACK's MOVE: ")
                print(board.board)
            else:
                print("No moves this turn")
                board.turn = not board.turn
        board.endGame()
        board.score()
    if board.board_score > 0:
        print("White won")
    else:
        print("Black won")

board = Board()
#play_minimax(board)
#play_MCTS(board)
#play_against_MCTS(board)
#play_against_minimax(board)
#MCTS_play_minimax(board)
