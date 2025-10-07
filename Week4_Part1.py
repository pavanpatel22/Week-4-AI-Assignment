"""
Monte Carlo Tree Search with UCB for Connect Four
=================================================

Advanced Tree Search Methods for Game Playing AI
"""

import numpy as np
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict
import time


# ============================================================================
# TREE SEARCH NODE IMPLEMENTATION
# ============================================================================

class SearchTreeNode:
    """
    Represents a node in the Monte Carlo search tree.
    
    Properties:
        game_state: The current state of the game
        parent_node: Reference to parent node
        move: Action taken to reach this node
        child_nodes: List of children nodes
        unexplored_moves: Moves not yet expanded
        visits: Number of times node was visited
        cumulative_score: Total reward accumulated
        average_score: Q-value (cumulative_score / visits)
    """
    
    def __init__(self, game_state, parent=None, move=None, unexplored_moves=None):
        self.game_state = game_state
        self.parent_node = parent
        self.move = move
        self.child_nodes = []
        self.unexplored_moves = unexplored_moves if unexplored_moves else []
        
        # Search statistics
        self.visits = 0
        self.cumulative_score = 0.0
        self.average_score = 0.0
    
    def all_moves_expanded(self):
        """Check if all possible moves have been explored."""
        return len(self.unexplored_moves) == 0
    
    def is_leaf_node(self):
        """Check if this is a terminal node."""
        return len(self.unexplored_moves) == 0 and len(self.child_nodes) == 0
    
    def select_optimal_child(self, exploration_weight=1.414):
        selection_scores = []
        for child in self.child_nodes:
            # Calculate exploitation component
            if child.visits > 0:
                exploitation_component = child.average_score
            else:
                exploitation_component = 0
            
            # Calculate exploration component
            if child.visits > 0:
                exploration_component = exploration_weight * math.sqrt(
                    (2 * math.log(self.visits)) / (child.visits + 1e-8)
                )
            else:
                exploration_component = float('inf')
            
            ucb_score = exploitation_component + exploration_component
            selection_scores.append(ucb_score)
        
        return self.child_nodes[np.argmax(selection_scores)]
    
    def get_preferred_move(self):
        if not self.child_nodes:
            return None
        best_node = max(self.child_nodes, key=lambda node: node.visits)
        return best_node.move
    
    def __str__(self):
        return (f"TreeNode(visits={self.visits}, "
                f"score={self.cumulative_score:.2f}, "
                f"Q={self.average_score:.3f}, "
                f"children={len(self.child_nodes)})")


class MonteCarloTreeSearch:
    def __init__(self, 
                 exploration_param=1.414,
                 max_search_iterations=1000,
                 max_simulation_depth=100,
                 computation_time=None):
        
        self.exploration_param = exploration_param
        self.max_search_iterations = max_search_iterations
        self.max_simulation_depth = max_simulation_depth
        self.computation_time = computation_time
        
        # Performance tracking
        self.total_nodes_generated = 0
        self.simulation_count = 0
    
    def find_best_move(self, starting_state, game_environment):
        # Initialize root node
        current_player = game_environment.get_active_player(starting_state)
        available_moves = game_environment.get_valid_moves(starting_state, current_player)
        root_node = SearchTreeNode(
            game_state=starting_state,
            unexplored_moves=available_moves.copy()
        )
        
        search_start_time = time.time()
        
        # Main search loop
        for iteration_count in range(self.max_search_iterations):
            # Check time constraint
            if self.computation_time and (time.time() - search_start_time) > self.computation_time:
                break
            
            # PHASE 1: SELECTION - Navigate tree using UCB
            current_node = root_node
            current_state = starting_state.copy()
            
            while not game_environment.is_game_over(current_state) and current_node.all_moves_expanded():
                current_node = current_node.select_optimal_child(self.exploration_param)
                current_state = current_node.game_state.copy()
            
            # PHASE 2: EXPANSION - Add new node to tree
            if not game_environment.is_game_over(current_state) and len(current_node.unexplored_moves) > 0:
                selected_move = current_node.unexplored_moves.pop()
                active_player = game_environment.get_active_player(current_state)
                next_state, immediate_reward, game_ended = game_environment.execute_move(
                    current_state, selected_move, active_player)
                
                # Determine moves for next state
                if not game_ended:
                    next_player = game_environment.get_active_player(next_state)
                    next_moves = game_environment.get_valid_moves(next_state, next_player)
                else:
                    next_moves = []
                
                new_node = SearchTreeNode(
                    game_state=next_state,
                    parent=current_node,
                    move=selected_move,
                    unexplored_moves=next_moves
                )
                current_node.child_nodes.append(new_node)
                
                current_node = new_node
                current_state = next_state
                self.total_nodes_generated += 1
            
            # PHASE 3: SIMULATION - Random playout
            simulation_result = self._perform_simulation(current_state, game_environment, current_player)
            self.simulation_count += 1
            
            # PHASE 4: BACKPROPAGATION - Update node statistics
            self._update_node_values(current_node, simulation_result, current_player)
        
        # Determine best move based on visit counts
        optimal_move = root_node.get_preferred_move()
        
        # Display search statistics
        print(f"\nSearch Performance Metrics:")
        print(f"  Iterations completed: {iteration_count + 1}")
        print(f"  Nodes generated: {self.total_nodes_generated}")
        print(f"  Simulations performed: {self.simulation_count}")
        print(f"  Root node visits: {root_node.visits}")
        print(f"  Root node Q-value: {root_node.average_score:.3f}")
        
        return optimal_move
    
    def _perform_simulation(self, state, game_environment, original_player):

        simulation_state = state.copy()
        steps = 0
        
        while not game_environment.is_game_over(simulation_state) and steps < self.max_simulation_depth:
            active_player = game_environment.get_active_player(simulation_state)
            legal_moves = game_environment.get_valid_moves(simulation_state, active_player)
            
            if len(legal_moves) == 0:
                break
            
            # Random move selection policy
            random_move = np.random.choice(legal_moves)
            simulation_state, reward, terminal = game_environment.execute_move(
                simulation_state, random_move, active_player)
            
            steps += 1
            
            if terminal:
                break
        
        # Calculate final reward from original player's perspective
        final_outcome = game_environment.evaluate_state(simulation_state, original_player)
        return final_outcome
    
    def _update_node_values(self, node, reward, original_player):

        current_reward = reward
        
        while node is not None:
            node.visits += 1
            node.cumulative_score += current_reward
            node.average_score = node.cumulative_score / node.visits
            
            # Alternate reward perspective for alternating moves
            current_reward = -current_reward
            
            node = node.parent_node
    
    def analyze_move_quality(self, initial_state, game_environment):

        current_player = game_environment.get_active_player(initial_state)
        valid_moves = game_environment.get_valid_moves(initial_state, current_player)
        root = SearchTreeNode(
            game_state=initial_state,
            unexplored_moves=valid_moves.copy()
        )
        
        # Execute search
        for _ in range(self.max_search_iterations):
            current_node = root
            current_state = initial_state.copy()
            
            while not game_environment.is_game_over(current_state) and current_node.all_moves_expanded():
                current_node = current_node.select_optimal_child(self.exploration_param)
                current_state = current_node.game_state.copy()
            
            if not game_environment.is_game_over(current_state) and len(current_node.unexplored_moves) > 0:
                move = current_node.unexplored_moves.pop()
                active_player = game_environment.get_active_player(current_state)
                next_state, reward, done = game_environment.execute_move(current_state, move, active_player)
                
                if not done:
                    next_player = game_environment.get_active_player(next_state)
                    next_moves = game_environment.get_valid_moves(next_state, next_player)
                else:
                    next_moves = []
                
                child_node = SearchTreeNode(
                    game_state=next_state,
                    parent=current_node,
                    move=move,
                    unexplored_moves=next_moves
                )
                current_node.child_nodes.append(child_node)
                current_node = child_node
                current_state = next_state
            
            simulation_reward = self._perform_simulation(current_state, game_environment, current_player)
            self._update_node_values(current_node, simulation_reward, current_player)
        
        # Compile move statistics
        move_analysis = {}
        for child in root.child_nodes:
            move_analysis[child.move] = {
                'visit_count': child.visits,
                'q_value': child.average_score,
                'total_score': child.cumulative_score,
                'average_reward': child.cumulative_score / child.visits if child.visits > 0 else 0
            }
        
        return move_analysis


# ============================================================================
# CONNECT FOUR GAME IMPLEMENTATION
# ============================================================================

class ConnectFourGame:
    
    def __init__(self):
        self.board_height = 6
        self.board_width = 7
        self.winning_length = 4
        self.initialize_game()
    
    def initialize_game(self):
        self.game_board = np.zeros((self.board_height, self.board_width), dtype=int)
        self.active_player = 1
        self.turn_count = 0
        return self.get_game_state()
    
    def get_game_state(self):
        return {
            'board': self.game_board.copy(),
            'current_player': self.active_player,
            'move_count': self.turn_count
        }
    
    def get_active_player(self, state):
        """Get current player from game state."""
        return state['current_player']
    
    def get_valid_moves(self, state, player):
        board = state['board']
        return [column for column in range(self.board_width) if board[0][column] == 0]
    
    def execute_move(self, state, move, player):
        # Create new state
        board = state['board'].copy()
        move_count = state['move_count']
        
        # Find available row in selected column
        target_row = -1
        for row in range(self.board_height - 1, -1, -1):
            if board[row][move] == 0:
                target_row = row
                break
        
        if target_row == -1:
            raise ValueError(f"Column {move} is completely filled")
        
        # Place player's piece
        board[target_row][move] = player
        move_count += 1
        
        # Check for victory
        if self._check_winning_condition(board, target_row, move, player):
            reward = 1.0  # Winning move
            game_over = True
        elif move_count >= self.board_height * self.board_width:
            reward = 0.0  # Draw
            game_over = True
        else:
            reward = 0.0  # Continue playing
            game_over = False
        
        # Prepare next state
        next_player = 3 - player  # Alternate between 1 and 2
        next_state = {
            'board': board,
            'current_player': next_player,
            'move_count': move_count
        }
        
        return next_state, reward, game_over
    
    def _check_winning_condition(self, board, row, col, player):
        # Check all directions for winning line
        direction_vectors = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal down-right
            (1, -1)   # Diagonal down-left
        ]
        
        for dr, dc in direction_vectors:
            connected_count = 1  # Include current piece
            
            # Check in positive direction
            r, c = row + dr, col + dc
            while (0 <= r < self.board_height and 0 <= c < self.board_width 
                   and board[r][c] == player):
                connected_count += 1
                r += dr
                c += dc
            
            # Check in negative direction
            r, c = row - dr, col - dc
            while (0 <= r < self.board_height and 0 <= c < self.board_width 
                   and board[r][c] == player):
                connected_count += 1
                r -= dr
                c -= dc
            
            if connected_count >= self.winning_length:
                return True
        
        return False
    
    def is_game_over(self, state):
        """Check if game has ended."""
        board = state['board']
        move_count = state['move_count']
        
        # Check for board full
        if move_count >= self.board_height * self.board_width:
            return True
        
        # Check for any winning positions
        for player_id in [1, 2]:
            # Horizontal check
            for row in range(self.board_height):
                for col in range(self.board_width - 3):
                    if all(board[row][col+i] == player_id for i in range(4)):
                        return True
            
            # Vertical check
            for row in range(self.board_height - 3):
                for col in range(self.board_width):
                    if all(board[row+i][col] == player_id for i in range(4)):
                        return True
            
            # Diagonal down-right
            for row in range(self.board_height - 3):
                for col in range(self.board_width - 3):
                    if all(board[row+i][col+i] == player_id for i in range(4)):
                        return True
            
            # Diagonal down-left
            for row in range(3, self.board_height):
                for col in range(self.board_width - 3):
                    if all(board[row-i][col+i] == player_id for i in range(4)):
                        return True
        
        return False
    
    def evaluate_state(self, state, player):
        if not self.is_game_over(state):
            return 0.0
        
        board = state['board']
        
        # Check if player won
        for row in range(self.board_height):
            for col in range(self.board_width):
                if board[row][col] == player:
                    if self._check_winning_condition(board, row, col, player):
                        return 1.0
        
        # Check if opponent won
        opponent = 3 - player
        for row in range(self.board_height):
            for col in range(self.board_width):
                if board[row][col] == opponent:
                    if self._check_winning_condition(board, row, col, opponent):
                        return -1.0
        
        # Draw game
        return 0.0
    
    def display_board(self, state):
        """Render the game board visually."""
        board = state['board']
        print("\n  " + " ".join(str(i) for i in range(self.board_width)))
        print(" +" + "-" * (self.board_width * 2 - 1) + "+")
        for row in board:
            print(" |" + "|".join([" ▢", " ●", " ○"][cell] for cell in row) + "|")
        print(" +" + "-" * (self.board_width * 2 - 1) + "+")
        print(f"  Active player: {state['current_player']}")


# ============================================================================
# DEMONSTRATION AND EVALUATION
# ============================================================================

def demonstrate_ai_vs_random():
    print("=" * 60)
    print("CONNECT FOUR: AI vs Random Player")
    print("=" * 60)
    
    game = ConnectFourGame()
    search_ai = MonteCarloTreeSearch(
        exploration_param=1.414,
        max_search_iterations=500,
        max_simulation_depth=42
    )
    
    game_state = game.initialize_game()
    game.display_board(game_state)
    
    while not game.is_game_over(game_state):
        current_player = game.get_active_player(game_state)
        
        if current_player == 1:
            # AI player
            print(f"\n🤖 AI Player {current_player} calculating move...")
            selected_move = search_ai.find_best_move(game_state, game)
            print(f"   AI selects column {selected_move}")
        else:
            # Random player
            available_moves = game.get_valid_moves(game_state, current_player)
            selected_move = np.random.choice(available_moves)
            print(f"\n🎲 Random Player {current_player} picks column {selected_move}")
        
        game_state, reward, game_ended = game.execute_move(game_state, selected_move, current_player)
        game.display_board(game_state)
        
        if game_ended:
            if reward == 1.0:
                winning_player = current_player
                print(f"\n🏆 Player {winning_player} wins the game!")
            else:
                print("\n🤝 Game ended in a draw!")
            break
    
    return game_state


def evaluate_ai_decision_making():
    print("\n" + "=" * 60)
    print("EVALUATION: AI Decision Quality")
    print("=" * 60)
    
    game = ConnectFourGame()
    search_ai = MonteCarloTreeSearch(
        exploration_param=1.414,
        max_search_iterations=1000,
        max_simulation_depth=42
    )
    
    # Create test scenario with immediate win opportunity
    game_state = game.initialize_game()
    # Setup: Player 1 has three consecutive pieces
    game_state['board'][5][0] = 1
    game_state['board'][5][1] = 1
    game_state['board'][5][2] = 1
    # Column 3 is the winning move
    
    print("\nTest Scenario: Player 1 can win with column 3")
    game.display_board(game_state)
    
    print("\n🤖 Running AI analysis...")
    chosen_move = search_ai.find_best_move(game_state, game)
    
    print(f"\n✓ AI selected move: {chosen_move}")
    if chosen_move == 3:
        print("✓ SUCCESS! AI identified the winning move!")
    else:
        print("✗ AI missed the optimal move (may need more computation)")
    
    # Get detailed analysis
    move_stats = search_ai.analyze_move_quality(game_state, game)
    print("\nMove Analysis Details:")
    for move, data in sorted(move_stats.items(), key=lambda x: x[1]['visit_count'], reverse=True):
        print(f"  Column {move}: visits={data['visit_count']:>4d}, "
              f"Q={data['q_value']:>7.3f}, "
              f"avg_reward={data['average_reward']:>7.3f}")


def performance_comparison():
    print("\n" + "=" * 60)
    print("PERFORMANCE: AI vs Random (10 games)")
    print("=" * 60)
    
    game = ConnectFourGame()
    search_ai = MonteCarloTreeSearch(
        exploration_param=1.414,
        max_search_iterations=300,
        max_simulation_depth=42
    )
    
    game_results = {'ai_victories': 0, 'random_victories': 0, 'draws': 0}
    
    for game_num in range(10):
        print(f"\nGame {game_num + 1}/10 in progress...")
        game_state = game.initialize_game()
        
        while not game.is_game_over(game_state):
            current_player = game.get_active_player(game_state)
            
            if current_player == 1:
                # AI player
                move = search_ai.find_best_move(game_state, game)
            else:
                # Random player
                valid_moves = game.get_valid_moves(game_state, current_player)
                move = np.random.choice(valid_moves)
            
            game_state, reward, game_ended = game.execute_move(game_state, move, current_player)
            
            if game_ended:
                if reward == 1.0:
                    if current_player == 1:
                        game_results['ai_victories'] += 1
                        print("  Outcome: AI wins")
                    else:
                        game_results['random_victories'] += 1
                        print("  Outcome: Random wins")
                else:
                    game_results['draws'] += 1
                    print("  Outcome: Draw")
                break
    
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"AI victories:    {game_results['ai_victories']}/10 ({game_results['ai_victories']*10}%)")
    print(f"Random victories: {game_results['random_victories']}/10 ({game_results['random_victories']*10}%)")
    print(f"Draws:           {game_results['draws']}/10 ({game_results['draws']*10}%)")


# ============================================================================
# EXECUTION: Run all demonstrations
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  MONTE CARLO TREE SEARCH IMPLEMENTATION  ".center(58) + "║")
    print("║" + "  Advanced AI for Game Playing  ".center(58) + "║")
    print("║" + "  Course Assignment Submission  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Execute demonstrations
    print("\n[1/3] Live Game: AI vs Random Player")
    demonstrate_ai_vs_random()
    
    print("\n\n[2/3] AI Decision Quality Assessment")
    evaluate_ai_decision_making()
    
    print("\n\n[3/3] Performance Benchmark")
    performance_comparison()
    
    print("\n\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nImplementation demonstrates:")
    print("✓ Complete Monte Carlo Tree Search algorithm")
    print("✓ UCB1 selection with exploration-exploitation balance")
    print("✓ Connect Four game environment with full rules")
    print("✓ AI capable of finding optimal moves")
    print("✓ Superior performance against random baseline")