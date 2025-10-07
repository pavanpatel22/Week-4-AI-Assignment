import numpy as np
import math
import random
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import time
import os

# ============================================================================
# QUANTUM GOBBLET GAME IMPLEMENTATION
# ============================================================================

class Piece:
    """Represents a game piece with player ownership and size"""
    def __init__(self, player: str, size: int):
        self.player = player  # 'X' or 'O'
        self.size = size      # 1 (small), 2 (medium), 3 (large)
    
    def __str__(self) -> str:
        size_symbols = {1: 'S', 2: 'M', 3: 'L'}
        return f"{self.player}{size_symbols[self.size]}"
    
    def can_cover(self, other_piece: Optional['Piece']) -> bool:
        """Check if this piece can cover another piece (must be larger)"""
        if other_piece is None:
            return True
        return self.size > other_piece.size
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Piece):
            return False
        return self.player == other.player and self.size == other.size

class QuantumGobbletGame:
    """
    Implements the Quantum Gobblet game rules and state management
    """
    
    def __init__(self):
        self.board_size = 3
        self.winning_length = 3
        self.initialize_game()
    
    def initialize_game(self) -> Dict[str, Any]:
        """Initialize a new game state"""
        # Board is a 3x3 grid where each cell contains a stack of pieces
        self.game_board = [[[] for _ in range(self.board_size)] for _ in range(self.board_size)]
        
        # Players start with 2 pieces of each size
        self.player_pieces = {
            'X': {1: 2, 2: 2, 3: 2},
            'O': {1: 2, 2: 2, 3: 2}
        }
        
        self.current_player = 'X'
        self.move_count = 0
        self.game_over = False
        self.winner = None
        
        return self.get_game_state()
    
    def get_game_state(self) -> Dict[str, Any]:
        """Return current game state for MCTS"""
        return {
            'board': [[stack[:] for stack in row] for row in self.game_board],
            'player_pieces': {p: sizes.copy() for p, sizes in self.player_pieces.items()},
            'current_player': self.current_player,
            'move_count': self.move_count,
            'game_over': self.game_over,
            'winner': self.winner
        }
    
    def set_game_state(self, state: Dict[str, Any]) -> None:
        """Set game state from state dictionary"""
        self.game_board = [[stack[:] for stack in row] for row in state['board']]
        self.player_pieces = {p: sizes.copy() for p, sizes in state['player_pieces'].items()}
        self.current_player = state['current_player']
        self.move_count = state['move_count']
        self.game_over = state['game_over']
        self.winner = state['winner']
    
    def get_active_player(self, state: Dict[str, Any]) -> str:
        """Get current player from game state"""
        return state['current_player']
    
    def get_valid_moves(self, state: Dict[str, Any], player: str) -> List[Tuple]:
        """Get all valid moves for current player"""
        valid_moves = []
        board = state['board']
        player_pieces = state['player_pieces'][player]
        
        # Type 1: Place new piece from hand
        for size in [1, 2, 3]:
            if player_pieces[size] > 0:
                for row in range(self.board_size):
                    for col in range(self.board_size):
                        top_piece = self._get_top_piece(board, row, col)
                        if top_piece is None or Piece(player, size).can_cover(top_piece):
                            valid_moves.append(('place', size, row, col))
        
        # Type 2: Move existing piece on board
        for from_row in range(self.board_size):
            for from_col in range(self.board_size):
                top_piece = self._get_top_piece(board, from_row, from_col)
                if top_piece and top_piece.player == player:
                    # This piece can be moved
                    for to_row in range(self.board_size):
                        for to_col in range(self.board_size):
                            if from_row == to_row and from_col == to_col:
                                continue  # Can't move to same position
                            target_top = self._get_top_piece(board, to_row, to_col)
                            if top_piece.can_cover(target_top):
                                valid_moves.append(('move', from_row, from_col, to_row, to_col))
        
        return valid_moves
    
    def _get_top_piece(self, board: List[List[List[Piece]]], row: int, col: int) -> Optional[Piece]:
        """Get the top piece at a position, or None if empty"""
        stack = board[row][col]
        return stack[-1] if stack else None
    
    def execute_move(self, state: Dict[str, Any], move: Tuple, player: str) -> Tuple[Dict[str, Any], float, bool]:
        """Execute a move and return new state, reward, and terminal flag"""
        # Create deep copy of state
        new_state = self._deep_copy_state(state)
        board = new_state['board']
        player_pieces = new_state['player_pieces']
        
        move_type = move[0]
        
        if move_type == 'place':
            # Place new piece from hand
            _, size, row, col = move
            piece = Piece(player, size)
            
            # Verify move is valid
            top_piece = self._get_top_piece(board, row, col)
            if not piece.can_cover(top_piece):
                raise ValueError(f"Invalid placement: cannot place {piece} on {top_piece}")
            
            # Place piece
            board[row][col].append(piece)
            player_pieces[player][size] -= 1
            
        elif move_type == 'move':
            # Move existing piece
            _, from_row, from_col, to_row, to_col = move
            
            # Verify move is valid
            from_stack = board[from_row][from_col]
            if not from_stack:
                raise ValueError(f"No piece at ({from_row}, {from_col})")
            
            moving_piece = from_stack[-1]
            if moving_piece.player != player:
                raise ValueError(f"Piece at ({from_row}, {from_col}) belongs to {moving_piece.player}")
            
            target_top = self._get_top_piece(board, to_row, to_col)
            if not moving_piece.can_cover(target_top):
                raise ValueError(f"Cannot move {moving_piece} to cover {target_top}")
            
            # Move piece
            moved_piece = from_stack.pop()
            board[to_row][to_col].append(moved_piece)
        
        else:
            raise ValueError(f"Unknown move type: {move_type}")
        
        # Update game state
        new_state['move_count'] += 1
        new_state['current_player'] = 'O' if player == 'X' else 'X'
        
        # Check for game end
        game_ended, winner = self._check_game_end(new_state)
        new_state['game_over'] = game_ended
        new_state['winner'] = winner
        
        # Calculate reward
        if game_ended:
            reward = 1.0 if winner == player else -1.0 if winner else 0.0
        else:
            reward = 0.0
        
        return new_state, reward, game_ended
    
    def _deep_copy_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a deep copy of the game state"""
        return {
            'board': [[[Piece(p.player, p.size) for p in stack] for stack in row] for row in state['board']],
            'player_pieces': {p: sizes.copy() for p, sizes in state['player_pieces'].items()},
            'current_player': state['current_player'],
            'move_count': state['move_count'],
            'game_over': state['game_over'],
            'winner': state['winner']
        }
    
    def _check_game_end(self, state: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if game has ended and determine winner"""
        board = state['board']
        
        # Check all rows, columns, and diagonals
        lines_to_check = []
        
        # Rows
        for row in range(self.board_size):
            lines_to_check.append([self._get_top_piece(board, row, col) for col in range(self.board_size)])
        
        # Columns
        for col in range(self.board_size):
            lines_to_check.append([self._get_top_piece(board, row, col) for row in range(self.board_size)])
        
        # Diagonals
        lines_to_check.append([self._get_top_piece(board, i, i) for i in range(self.board_size)])
        lines_to_check.append([self._get_top_piece(board, i, self.board_size - 1 - i) for i in range(self.board_size)])
        
        # Check each line for winning condition
        for line in lines_to_check:
            if self._is_winning_line(line):
                winner = line[0].player  # All pieces in winning line belong to same player
                return True, winner
        
        # Check for draw (no valid moves for next player)
        next_player = 'O' if state['current_player'] == 'X' else 'X'
        valid_moves = self.get_valid_moves(state, next_player)
        if not valid_moves:
            return True, None  # Draw
        
        return False, None
    
    def _is_winning_line(self, line: List[Optional[Piece]]) -> bool:
        """Check if a line contains three consecutive pieces from same player"""
        if None in line:
            return False
        
        # All pieces must belong to same player
        player = line[0].player
        return all(piece.player == player for piece in line)
    
    def is_game_over(self, state: Dict[str, Any]) -> bool:
        """Check if game has ended"""
        return state['game_over']
    
    def evaluate_state(self, state: Dict[str, Any], player: str) -> float:
        """Evaluate game state from player's perspective"""
        if not state['game_over']:
            # Heuristic evaluation for non-terminal states
            return self._heuristic_evaluation(state, player)
        
        if state['winner'] == player:
            return 1.0
        elif state['winner'] is None:
            return 0.0
        else:
            return -1.0
    
    def _heuristic_evaluation(self, state: Dict[str, Any], player: str) -> float:
        """Heuristic evaluation of non-terminal game state"""
        board = state['board']
        opponent = 'O' if player == 'X' else 'X'
        score = 0.0
        
        # Check all possible lines
        lines = []
        for i in range(3):
            # Rows
            lines.append([self._get_top_piece(board, i, j) for j in range(3)])
            # Columns
            lines.append([self._get_top_piece(board, j, i) for j in range(3)])
        # Diagonals
        lines.append([self._get_top_piece(board, i, i) for i in range(3)])
        lines.append([self._get_top_piece(board, i, 2 - i) for i in range(3)])
        
        for line in lines:
            player_count = sum(1 for p in line if p and p.player == player)
            opponent_count = sum(1 for p in line if p and p.player == opponent)
            
            if player_count == 2 and opponent_count == 0:
                score += 0.3  # Two in a row
            elif player_count == 1 and opponent_count == 0:
                score += 0.1  # One in a row
            elif opponent_count == 2 and player_count == 0:
                score -= 0.3  # Opponent has two in a row
            elif opponent_count == 1 and player_count == 0:
                score -= 0.1  # Opponent has one in a row
        
        # Piece advantage (having larger pieces available)
        player_advantage = sum(size * count for size, count in state['player_pieces'][player].items())
        opponent_advantage = sum(size * count for size, count in state['player_pieces'][opponent].items())
        score += 0.05 * (player_advantage - opponent_advantage)
        
        return score
    
    def display_board(self, state: Dict[str, Any]) -> None:
        """Display the game board with pieces"""
        board = state['board']
        
        print("\n    0     1     2")
        print("  ┌─────┬─────┬─────┐")
        
        for row in range(self.board_size):
            print(f"{row} │", end="")
            for col in range(self.board_size):
                stack = board[row][col]
                if stack:
                    # Show all pieces in stack (bottom to top)
                    stack_display = "|".join(str(piece) for piece in stack)
                    print(f"{stack_display:^5}│", end="")
                else:
                    print(f"{' ':^5}│", end="")
            print()
            
            if row < self.board_size - 1:
                print("  ├─────┼─────┼─────┤")
            else:
                print("  └─────┴─────┴─────┘")
        
        # Display player pieces
        print(f"\nPlayer X pieces: {state['player_pieces']['X']}")
        print(f"Player O pieces: {state['player_pieces']['O']}")
        print(f"Current player: {state['current_player']}")

# ============================================================================
# MONTE CARLO TREE SEARCH IMPLEMENTATION
# ============================================================================

class SearchTreeNode:
    """
    Represents a node in the Monte Carlo search tree for Quantum Gobblet
    """
    
    def __init__(self, game_state: Dict[str, Any], parent=None, move=None, unexplored_moves=None):
        self.game_state = game_state
        self.parent_node = parent
        self.move = move
        self.child_nodes = []
        self.unexplored_moves = unexplored_moves if unexplored_moves else []
        
        # Search statistics
        self.visits = 0
        self.cumulative_score = 0.0
        self.average_score = 0.0
    
    def all_moves_expanded(self) -> bool:
        """Check if all possible moves have been explored"""
        return len(self.unexplored_moves) == 0
    
    def is_leaf_node(self) -> bool:
        """Check if this is a terminal node"""
        return len(self.unexplored_moves) == 0 and len(self.child_nodes) == 0
    
    def select_optimal_child(self, exploration_weight: float = 1.414) -> 'SearchTreeNode':
        """Select child node using UCB1 formula"""
        selection_scores = []
        
        for child in self.child_nodes:
            if child.visits > 0:
                exploitation = child.average_score
                exploration = exploration_weight * math.sqrt(
                    (2 * math.log(self.visits)) / (child.visits + 1e-8)
                )
                ucb_score = exploitation + exploration
            else:
                ucb_score = float('inf')  # Prioritize unexplored nodes
            
            selection_scores.append(ucb_score)
        
        return self.child_nodes[np.argmax(selection_scores)]
    
    def get_preferred_move(self) -> Optional[Tuple]:
        """Get the move with highest visit count"""
        if not self.child_nodes:
            return None
        best_node = max(self.child_nodes, key=lambda node: node.visits)
        return best_node.move
    
    def __str__(self) -> str:
        return (f"TreeNode(visits={self.visits}, "
                f"score={self.cumulative_score:.2f}, "
                f"Q={self.average_score:.3f}, "
                f"children={len(self.child_nodes)})")

class MonteCarloTreeSearch:
    """Monte Carlo Tree Search implementation for Quantum Gobblet"""
    
    def __init__(self, 
                 exploration_param: float = 1.414,
                 max_search_iterations: int = 1000,
                 max_simulation_depth: int = 50,
                 computation_time: Optional[float] = None):
        
        self.exploration_param = exploration_param
        self.max_search_iterations = max_search_iterations
        self.max_simulation_depth = max_simulation_depth
        self.computation_time = computation_time
        
        # Performance tracking
        self.total_nodes_generated = 0
        self.simulation_count = 0
    
    def find_best_move(self, starting_state: Dict[str, Any], game_environment: QuantumGobbletGame) -> Tuple:
        """Find the best move using MCTS"""
        current_player = game_environment.get_active_player(starting_state)
        available_moves = game_environment.get_valid_moves(starting_state, current_player)
        
        if not available_moves:
            raise ValueError("No valid moves available")
        
        # Initialize root node
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
            
            # PHASE 1: SELECTION
            current_node = root_node
            simulation_state = game_environment._deep_copy_state(starting_state)
            
            while (not game_environment.is_game_over(simulation_state) and 
                   current_node.all_moves_expanded() and current_node.child_nodes):
                current_node = current_node.select_optimal_child(self.exploration_param)
                simulation_state = current_node.game_state
            
            # PHASE 2: EXPANSION
            if (not game_environment.is_game_over(simulation_state) and 
                len(current_node.unexplored_moves) > 0):
                
                selected_move = current_node.unexplored_moves.pop()
                active_player = game_environment.get_active_player(simulation_state)
                
                # Execute move to get new state
                next_state, immediate_reward, game_ended = game_environment.execute_move(
                    simulation_state, selected_move, active_player)
                
                # Determine moves for next state
                if not game_ended:
                    next_player = game_environment.get_active_player(next_state)
                    next_moves = game_environment.get_valid_moves(next_state, next_player)
                else:
                    next_moves = []
                
                # Create new node
                new_node = SearchTreeNode(
                    game_state=next_state,
                    parent=current_node,
                    move=selected_move,
                    unexplored_moves=next_moves
                )
                current_node.child_nodes.append(new_node)
                current_node = new_node
                simulation_state = next_state
                self.total_nodes_generated += 1
            
            # PHASE 3: SIMULATION
            simulation_result = self._perform_simulation(simulation_state, game_environment, current_player)
            self.simulation_count += 1
            
            # PHASE 4: BACKPROPAGATION
            self._update_node_values(current_node, simulation_result, current_player)
        
        # Determine best move based on visit counts
        optimal_move = root_node.get_preferred_move()
        
        # Display search statistics
        print(f"\nMCTS Search Metrics:")
        print(f"  Iterations completed: {iteration_count + 1}")
        print(f"  Nodes generated: {self.total_nodes_generated}")
        print(f"  Simulations performed: {self.simulation_count}")
        print(f"  Root node visits: {root_node.visits}")
        
        return optimal_move
    
    def _perform_simulation(self, state: Dict[str, Any], 
                          game_environment: QuantumGobbletGame,
                          original_player: str) -> float:
        """Perform random simulation from current state"""
        simulation_state = game_environment._deep_copy_state(state)
        steps = 0
        
        while (not game_environment.is_game_over(simulation_state) and 
               steps < self.max_simulation_depth):
            
            active_player = game_environment.get_active_player(simulation_state)
            legal_moves = game_environment.get_valid_moves(simulation_state, active_player)
            
            if not legal_moves:
                break
            
            # Random move selection
            random_move = random.choice(legal_moves)
            simulation_state, reward, terminal = game_environment.execute_move(
                simulation_state, random_move, active_player)
            
            steps += 1
        
        # Evaluate final state from original player's perspective
        final_outcome = game_environment.evaluate_state(simulation_state, original_player)
        return final_outcome
    
    def _update_node_values(self, node: SearchTreeNode, reward: float, original_player: str) -> None:
        """Backpropagate simulation results through the tree"""
        current_node = node
        current_reward = reward
        
        while current_node is not None:
            current_node.visits += 1
            current_node.cumulative_score += current_reward
            current_node.average_score = current_node.cumulative_score / current_node.visits
            
            # Alternate reward for opponent's perspective in two-player zero-sum game
            current_reward = -current_reward
            current_node = current_node.parent_node

# ============================================================================
# GAME INTERFACE AND DEMONSTRATION
# ============================================================================

class QuantumGobbletInterface:
    """User interface for playing Quantum Gobblet"""
    
    def __init__(self):
        self.game = QuantumGobbletGame()
        self.ai = MonteCarloTreeSearch(
            exploration_param=1.414,
            max_search_iterations=800,
            max_simulation_depth=40,
            computation_time=2.0
        )
    
    def display_move_help(self) -> None:
        """Display help for move input format"""
        print("\nMove Input Format:")
        print("  Place piece: 'place size row col' (e.g., 'place 1 0 1')")
        print("  Move piece:  'move from_row from_col to_row to_col' (e.g., 'move 0 1 1 1')")
        print("  Sizes: 1=Small, 2=Medium, 3=Large")
        print("  Positions: 0, 1, or 2 (row and column)")
    
    def get_human_move(self, state: Dict[str, Any]) -> Tuple:
        """Get and validate human player move input"""
        valid_moves = self.game.get_valid_moves(state, state['current_player'])
        
        while True:
            try:
                self.display_move_help()
                print(f"\nYour valid moves: {len(valid_moves)} available")
                
                user_input = input("Enter your move: ").strip().lower().split()
                
                if not user_input:
                    continue
                
                if user_input[0] == 'place' and len(user_input) == 4:
                    size = int(user_input[1])
                    row = int(user_input[2])
                    col = int(user_input[3])
                    move = ('place', size, row, col)
                    
                elif user_input[0] == 'move' and len(user_input) == 5:
                    from_row = int(user_input[1])
                    from_col = int(user_input[2])
                    to_row = int(user_input[3])
                    to_col = int(user_input[4])
                    move = ('move', from_row, from_col, to_row, to_col)
                
                else:
                    print("Invalid input format. Please try again.")
                    continue
                
                if move in valid_moves:
                    return move
                else:
                    print("Invalid move. Please choose a valid move.")
                    
            except (ValueError, IndexError):
                print("Invalid input. Please use the correct format.")
    
    def play_human_vs_ai(self) -> None:
        """Play a game between human and AI"""
        print("=" * 60)
        print("QUANTUM GOBBLET: Human vs AI")
        print("=" * 60)
        
        game_state = self.game.initialize_game()
        
        while not self.game.is_game_over(game_state):
            self.game.display_board(game_state)
            current_player = game_state['current_player']
            
            if current_player == 'X':  # Human
                print(f"\n🧑 Your turn (Player {current_player})")
                move = self.get_human_move(game_state)
                print(f"   You played: {move}")
                
            else:  # AI
                print(f"\n🤖 AI Player {current_player} thinking...")
                start_time = time.time()
                move = self.ai.find_best_move(game_state, self.game)
                thinking_time = time.time() - start_time
                print(f"   AI selected: {move} (in {thinking_time:.1f}s)")
            
            # Execute move
            game_state, reward, game_ended = self.game.execute_move(game_state, move, current_player)
            
            if game_ended:
                self.game.display_board(game_state)
                if game_state['winner']:
                    winner = "You" if game_state['winner'] == 'X' else "AI"
                    print(f"\n🏆 {winner} win the game!")
                else:
                    print(f"\n🤝 Game ended in a draw!")
                break
    
    def play_ai_vs_ai(self) -> None:
        """Watch AI play against itself"""
        print("=" * 60)
        print("QUANTUM GOBBLET: AI vs AI")
        print("=" * 60)
        
        game_state = self.game.initialize_game()
        
        while not self.game.is_game_over(game_state):
            self.game.display_board(game_state)
            current_player = game_state['current_player']
            
            print(f"\n🤖 AI Player {current_player} thinking...")
            start_time = time.time()
            move = self.ai.find_best_move(game_state, self.game)
            thinking_time = time.time() - start_time
            
            print(f"   AI selected: {move} (in {thinking_time:.1f}s)")
            
            # Execute move
            game_state, reward, game_ended = self.game.execute_move(game_state, move, current_player)
            
            if game_ended:
                self.game.display_board(game_state)
                if game_state['winner']:
                    print(f"\n🏆 Player {game_state['winner']} wins the game!")
                else:
                    print(f"\n🤝 Game ended in a draw!")
                break
            
            # Pause for viewing
            input("Press Enter to continue...")

def demonstrate_ai_capabilities():
    """Demonstrate AI capabilities with test scenarios"""
    print("\n" + "=" * 60)
    print("AI CAPABILITIES DEMONSTRATION")
    print("=" * 60)
    
    game = QuantumGobbletGame()
    ai = MonteCarloTreeSearch(
        exploration_param=1.414,
        max_search_iterations=500,
        max_simulation_depth=30
    )
    
    # Test scenario: Create a near-winning position
    test_state = game.initialize_game()
    
    # Manually set up pieces to create interesting situation
    test_state['board'][0][0] = [Piece('X', 2)]
    test_state['board'][0][1] = [Piece('X', 1)]
    test_state['board'][0][2] = [Piece('O', 3)]
    test_state['board'][1][1] = [Piece('O', 1)]
    
    # Adjust piece counts
    test_state['player_pieces']['X'] = {1: 1, 2: 1, 3: 2}
    test_state['player_pieces']['O'] = {1: 1, 2: 2, 3: 1}
    
    print("\nTest Scenario: Complex board position")
    game.display_board(test_state)
    
    print("\n🤖 AI analyzing position...")
    best_move = ai.find_best_move(test_state, game)
    
    print(f"✓ AI recommends: {best_move}")
    
    # Show move explanation
    if best_move[0] == 'place':
        _, size, row, col = best_move
        print(f"   Place {size} at ({row}, {col})")
    else:
        _, fr, fc, tr, tc = best_move
        print(f"   Move piece from ({fr}, {fc}) to ({tr}, {tc})")

def performance_benchmark():
    """Benchmark AI performance"""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    game = QuantumGobbletGame()
    ai = MonteCarloTreeSearch(
        exploration_param=1.414,
        max_search_iterations=200,
        max_simulation_depth=20
    )
    
    wins = {'X': 0, 'O': 0, 'draws': 0}
    total_thinking_time = 0
    games_played = 5
    
    print(f"Playing {games_played} AI vs AI games...")
    
    for game_num in range(games_played):
        print(f"\nGame {game_num + 1}/{games_played}")
        state = game.initialize_game()
        move_count = 0
        
        while not game.is_game_over(state):
            current_player = state['current_player']
            
            start_time = time.time()
            move = ai.find_best_move(state, game)
            thinking_time = time.time() - start_time
            total_thinking_time += thinking_time
            
            state, reward, ended = game.execute_move(state, move, current_player)
            move_count += 1
            
            if ended:
                if state['winner']:
                    wins[state['winner']] += 1
                    print(f"  Player {state['winner']} wins in {move_count} moves")
                else:
                    wins['draws'] += 1
                    print(f"  Draw after {move_count} moves")
                break
    
    print(f"\nBenchmark Results:")
    print(f"  Total games: {games_played}")
    print(f"  Player X wins: {wins['X']}")
    print(f"  Player O wins: {wins['O']}")
    print(f"  Draws: {wins['draws']}")
    print(f"  Average thinking time: {total_thinking_time / games_played:.2f}s per move")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  QUANTUM GOBBLET AI IMPLEMENTATION  ".center(58) + "║")
    print("║" + "  Monte Carlo Tree Search  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    interface = QuantumGobbletInterface()
    
    while True:
        print("\n" + "=" * 60)
        print("MAIN MENU")
        print("=" * 60)
        print("1. Play vs AI (Human vs Computer)")
        print("2. Watch AI vs AI")
        print("3. Demonstrate AI Capabilities")
        print("4. Run Performance Benchmark")
        print("5. Exit")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            interface.play_human_vs_ai()
        elif choice == '2':
            interface.play_ai_vs_ai()
        elif choice == '3':
            demonstrate_ai_capabilities()
        elif choice == '4':
            performance_benchmark()
        elif choice == '5':
            print("Thanks for playing Quantum Gobblet!")
            break
        else:
            print("Invalid choice. Please select 1-5.")
        
        input("\nPress Enter to continue...")