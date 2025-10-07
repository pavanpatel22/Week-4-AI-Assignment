"""
Part 1: Core MCTS-UCT Algorithm with Connect 4 Environment
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Any
from collections import defaultdict
from copy import deepcopy
import time


# ============================================================================
# PART 1A: MCTS-UCT Core Implementation
# ============================================================================

class MCTSNode:
    """
    Node in the MCTS search tree.
    
    Attributes:
        state: Current game/environment state
        parent: Parent node in tree
        action: Action that led to this node from parent
        children: List of child nodes
        untried_actions: Actions not yet explored from this node
        visit_count: N(s) - number of times node visited
        total_reward: R(s) - cumulative reward through this node
        q_value: Q(s,a) - average reward (total_reward / visit_count)
    """
    
    def __init__(self, state, parent=None, action=None, untried_actions=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions = untried_actions if untried_actions else []
        
        # Statistics for UCT
        self.visit_count = 0
        self.total_reward = 0.0
        self.q_value = 0.0
    
    def is_fully_expanded(self):
        """Check if all actions from this node have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self):
        """Check if this is a terminal state."""
        return len(self.untried_actions) == 0 and len(self.children) == 0
    
    def best_child(self, c_param=1.414):
        choices_weights = []
        for child in self.children:
            # Exploitation term: average reward
            exploitation = child.q_value / child.visit_count if child.visit_count > 0 else 0
            
            # Exploration term: UCB bonus
            exploration = c_param * math.sqrt(
                (2 * math.log(self.visit_count)) / (child.visit_count + 1e-10)
            )
            
            uct_value = exploitation + exploration
            choices_weights.append(uct_value)
        
        return self.children[np.argmax(choices_weights)]
    
    def best_action(self):
        if not self.children:
            return None
        best_child = max(self.children, key=lambda c: c.visit_count)
        return best_child.action
    
    def __repr__(self):
        return (f"MCTSNode(visits={self.visit_count}, "
                f"reward={self.total_reward:.2f}, "
                f"Q={self.q_value:.3f}, "
                f"children={len(self.children)})")


class MCTS_UCT:
    def __init__(self, 
                 exploration_constant=1.414,
                 max_iterations=1000,
                 max_rollout_depth=100,
                 time_limit=None):
        
        self.c = exploration_constant
        self.max_iterations = max_iterations
        self.max_rollout_depth = max_rollout_depth
        self.time_limit = time_limit
        
        # Statistics tracking
        self.nodes_explored = 0
        self.total_simulations = 0
    
    def search(self, initial_state, environment):
        # Initialize root node
        player = environment.get_current_player(initial_state)
        legal_actions = environment.get_legal_actions(initial_state, player)
        root = MCTSNode(
            state=initial_state,
            untried_actions=legal_actions.copy()
        )
        
        start_time = time.time()
        
        # Main MCTS loop
        for iteration in range(self.max_iterations):
            # Check time limit
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break
            
            # 1. SELECTION: Traverse tree using UCT
            node = root
            state = deepcopy(initial_state)
            
            while not environment.is_terminal(state) and node.is_fully_expanded():
                node = node.best_child(self.c)
                state = deepcopy(node.state)
            
            # 2. EXPANSION: Add one child node
            if not environment.is_terminal(state) and len(node.untried_actions) > 0:
                action = node.untried_actions.pop()
                current_player = environment.get_current_player(state)
                next_state, reward, done = environment.step(state, action, current_player)
                
                # Get legal actions for new state
                if not done:
                    next_player = environment.get_current_player(next_state)
                    next_actions = environment.get_legal_actions(next_state, next_player)
                else:
                    next_actions = []
                
                child_node = MCTSNode(
                    state=next_state,
                    parent=node,
                    action=action,
                    untried_actions=next_actions
                )
                node.children.append(child_node)
                
                node = child_node
                state = next_state
                self.nodes_explored += 1
            
            # 3. SIMULATION: Rollout with default policy
            simulation_reward = self._simulate(state, environment, player)
            self.total_simulations += 1
            
            # 4. BACKPROPAGATION: Update statistics
            self._backpropagate(node, simulation_reward, player)
        
        # Return best action based on visit counts
        best_action = root.best_action()
        
        # Print statistics
        print(f"\nMCTS Statistics:")
        print(f"  Iterations: {iteration + 1}")
        print(f"  Nodes explored: {self.nodes_explored}")
        print(f"  Total simulations: {self.total_simulations}")
        print(f"  Root visits: {root.visit_count}")
        print(f"  Root Q-value: {root.q_value:.3f}")
        
        return best_action
    
    def _simulate(self, state, environment, original_player):

        state = deepcopy(state)
        depth = 0
        
        while not environment.is_terminal(state) and depth < self.max_rollout_depth:
            current_player = environment.get_current_player(state)
            legal_actions = environment.get_legal_actions(state, current_player)
            
            if len(legal_actions) == 0:
                break
            
            # Default policy: random action
            action = np.random.choice(legal_actions)
            state, reward, done = environment.step(state, action, current_player)
            
            depth += 1
            
            if done:
                break
        
        # Get final reward from original player's perspective
        final_reward = environment.get_reward(state, original_player)
        return final_reward
    
    def _backpropagate(self, node, reward, original_player):

        current_reward = reward
        
        while node is not None:
            node.visit_count += 1
            node.total_reward += current_reward
            node.q_value = node.total_reward / node.visit_count
            
            # For two-player games: alternate perspective
            current_reward = -current_reward
            
            node = node.parent
    
    def get_action_statistics(self, initial_state, environment):

        player = environment.get_current_player(initial_state)
        legal_actions = environment.get_legal_actions(initial_state, player)
        root = MCTSNode(
            state=initial_state,
            untried_actions=legal_actions.copy()
        )
        
        # Run search
        for _ in range(self.max_iterations):
            node = root
            state = deepcopy(initial_state)
            
            while not environment.is_terminal(state) and node.is_fully_expanded():
                node = node.best_child(self.c)
                state = deepcopy(node.state)
            
            if not environment.is_terminal(state) and len(node.untried_actions) > 0:
                action = node.untried_actions.pop()
                current_player = environment.get_current_player(state)
                next_state, reward, done = environment.step(state, action, current_player)
                
                if not done:
                    next_player = environment.get_current_player(next_state)
                    next_actions = environment.get_legal_actions(next_state, next_player)
                else:
                    next_actions = []
                
                child_node = MCTSNode(
                    state=next_state,
                    parent=node,
                    action=action,
                    untried_actions=next_actions
                )
                node.children.append(child_node)
                node = child_node
                state = next_state
            
            simulation_reward = self._simulate(state, environment, player)
            self._backpropagate(node, simulation_reward, player)
        
        # Collect statistics
        stats = {}
        for child in root.children:
            stats[child.action] = {
                'visits': child.visit_count,
                'q_value': child.q_value,
                'total_reward': child.total_reward,
                'mean_reward': child.total_reward / child.visit_count if child.visit_count > 0 else 0
            }
        
        return stats


# ============================================================================
# PART 1B: Connect 4 Environment
# ============================================================================

class Connect4:
    
    def __init__(self):
        self.rows = 6
        self.cols = 7
        self.connect = 4
        self.reset()
    
    def reset(self):
        self.board = np.zeros((self.rows, self.cols), dtype=int)
        self.current_player = 1
        self.move_count = 0
        return self.get_state()
    
    def get_state(self):
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'move_count': self.move_count
        }
    
    def get_current_player(self, state):
        """Get current player from state."""
        return state['current_player']
    
    def get_legal_actions(self, state, player):
        board = state['board']
        return [col for col in range(self.cols) if board[0][col] == 0]
    
    def step(self, state, action, player):
        # Create new state
        board = state['board'].copy()
        move_count = state['move_count']
        
        # Find lowest empty row in column
        row = -1
        for r in range(self.rows - 1, -1, -1):
            if board[r][action] == 0:
                row = r
                break
        
        if row == -1:
            raise ValueError(f"Invalid move: column {action} is full")
        
        # Place piece
        board[row][action] = player
        move_count += 1
        
        # Check for win
        if self._check_win(board, row, action, player):
            reward = 1.0  # Win
            done = True
        elif move_count >= self.rows * self.cols:
            reward = 0.0  # Draw
            done = True
        else:
            reward = 0.0  # Game continues
            done = False
        
        # Create next state
        next_player = 3 - player  # Switch between 1 and 2
        next_state = {
            'board': board,
            'current_player': next_player,
            'move_count': move_count
        }
        
        return next_state, reward, done
    
    def _check_win(self, board, row, col, player):
        directions = [
            (0, 1),   # Horizontal
            (1, 0),   # Vertical
            (1, 1),   # Diagonal \
            (1, -1)   # Diagonal /
        ]
        
        for dr, dc in directions:
            count = 1  # Count the piece we just placed
            
            # Check positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.rows and 0 <= c < self.cols and board[r][c] == player:
                count += 1
                r += dr
                c += dc
            
            # Check negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.rows and 0 <= c < self.cols and board[r][c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= self.connect:
                return True
        
        return False
    
    def is_terminal(self, state):
        """Check if state is terminal (game over)."""
        board = state['board']
        move_count = state['move_count']
        
        # Check if board is full
        if move_count >= self.rows * self.cols:
            return True
        
        # Check if either player has won
        # (Quick check: look for any 4-in-a-row)
        for player in [1, 2]:
            # Check rows
            for row in range(self.rows):
                for col in range(self.cols - 3):
                    if all(board[row][col+i] == player for i in range(4)):
                        return True
            
            # Check columns
            for row in range(self.rows - 3):
                for col in range(self.cols):
                    if all(board[row+i][col] == player for i in range(4)):
                        return True
            
            # Check diagonals (\)
            for row in range(self.rows - 3):
                for col in range(self.cols - 3):
                    if all(board[row+i][col+i] == player for i in range(4)):
                        return True
            
            # Check diagonals (/)
            for row in range(3, self.rows):
                for col in range(self.cols - 3):
                    if all(board[row-i][col+i] == player for i in range(4)):
                        return True
        
        return False
    
    def get_reward(self, state, player):
        if not self.is_terminal(state):
            return 0.0
        
        board = state['board']
        
        # Check if player won
        for row in range(self.rows):
            for col in range(self.cols):
                if board[row][col] == player:
                    if self._check_win(board, row, col, player):
                        return 1.0
        
        # Check if opponent won
        opponent = 3 - player
        for row in range(self.rows):
            for col in range(self.cols):
                if board[row][col] == opponent:
                    if self._check_win(board, row, col, opponent):
                        return -1.0
        
        # Draw
        return 0.0
    
    def render(self, state):
        """Print the board in a human-readable format."""
        board = state['board']
        print("\n  0 1 2 3 4 5 6")
        print(" +" + "-" * (self.cols * 2 - 1) + "+")
        for row in board:
            print(" |" + "|".join([".⚫⚪"[cell] for cell in row]) + "|")
        print(" +" + "-" * (self.cols * 2 - 1) + "+")
        print(f"  Current player: {state['current_player']}")


# ============================================================================
# PART 1C: Demo and Testing
# ============================================================================

def play_game_mcts_vs_random():
    print("=" * 60)
    print("CONNECT 4: MCTS-UCT vs Random Player")
    print("=" * 60)
    
    env = Connect4()
    mcts = MCTS_UCT(
        exploration_constant=1.414,
        max_iterations=500,  # Adjust based on desired thinking time
        max_rollout_depth=42
    )
    
    state = env.reset()
    env.render(state)
    
    while not env.is_terminal(state):
        current_player = env.get_current_player(state)
        
        if current_player == 1:
            # MCTS player
            print(f"\n🤖 MCTS Player {current_player} thinking...")
            action = mcts.search(state, env)
            print(f"   MCTS chooses column {action}")
        else:
            # Random player
            legal_actions = env.get_legal_actions(state, current_player)
            action = np.random.choice(legal_actions)
            print(f"\n🎲 Random Player {current_player} chooses column {action}")
        
        state, reward, done = env.step(state, action, current_player)
        env.render(state)
        
        if done:
            if reward == 1.0:
                winner = current_player
                print(f"\n🎉 Player {winner} wins!")
            else:
                print("\n🤝 It's a draw!")
            break
    
    return state


def test_mcts_decision_quality():
    print("\n" + "=" * 60)
    print("TEST: MCTS Decision Quality")
    print("=" * 60)
    
    env = Connect4()
    mcts = MCTS_UCT(
        exploration_constant=1.414,
        max_iterations=1000,
        max_rollout_depth=42
    )
    
    # Create a position where Player 1 can win immediately
    state = env.reset()
    # Set up a board where Player 1 has 3 in a row horizontally
    state['board'][5][0] = 1
    state['board'][5][1] = 1
    state['board'][5][2] = 1
    # Column 3 is the winning move
    
    print("\nTest Position: Player 1 can win by playing column 3")
    env.render(state)
    
    print("\n🤖 Running MCTS...")
    action = mcts.search(state, env)
    
    print(f"\n✓ MCTS selected action: {action}")
    if action == 3:
        print("✓ CORRECT! MCTS found the winning move!")
    else:
        print("✗ MCTS missed the winning move (may need more iterations)")
    
    # Get detailed statistics
    stats = mcts.get_action_statistics(state, env)
    print("\nAction Statistics:")
    for action, data in sorted(stats.items(), key=lambda x: x[1]['visits'], reverse=True):
        print(f"  Column {action}: visits={data['visits']:>4d}, "
              f"Q={data['q_value']:>7.3f}, "
              f"mean_reward={data['mean_reward']:>7.3f}")


def benchmark_mcts():
    print("\n" + "=" * 60)
    print("BENCHMARK: MCTS vs Random (10 games)")
    print("=" * 60)
    
    env = Connect4()
    mcts = MCTS_UCT(
        exploration_constant=1.414,
        max_iterations=300,
        max_rollout_depth=42
    )
    
    results = {'mcts_wins': 0, 'random_wins': 0, 'draws': 0}
    
    for game in range(10):
        print(f"\nGame {game + 1}/10...")
        state = env.reset()
        
        while not env.is_terminal(state):
            current_player = env.get_current_player(state)
            
            if current_player == 1:
                # MCTS player
                action = mcts.search(state, env)
            else:
                # Random player
                legal_actions = env.get_legal_actions(state, current_player)
                action = np.random.choice(legal_actions)
            
            state, reward, done = env.step(state, action, current_player)
            
            if done:
                if reward == 1.0:
                    if current_player == 1:
                        results['mcts_wins'] += 1
                        print("  Result: MCTS wins")
                    else:
                        results['random_wins'] += 1
                        print("  Result: Random wins")
                else:
                    results['draws'] += 1
                    print("  Result: Draw")
                break
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"MCTS wins:   {results['mcts_wins']}/10 ({results['mcts_wins']*10}%)")
    print(f"Random wins: {results['random_wins']}/10 ({results['random_wins']*10}%)")
    print(f"Draws:       {results['draws']}/10 ({results['draws']*10}%)")


# ============================================================================
# MAIN: Run all demonstrations
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  MCTS-UCT IMPLEMENTATION WITH CONNECT 4  ".center(58) + "║")
    print("║" + "  Week 4 Assignment.     ".center(58) + "║")
    print("║" + "  Northeastern University Vancouver       ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # Run demonstrations
    print("\n[1/3] Playing a full game: MCTS vs Random")
    play_game_mcts_vs_random()
    
    print("\n\n[2/3] Testing MCTS decision quality")
    test_mcts_decision_quality()
    
    print("\n\n[3/3] Running benchmark")
    benchmark_mcts()
    
    print("\n\n" + "=" * 60)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 60)
    print("\nThis implementation demonstrates:")
    print("Complete MCTS-UCT algorithm (4 phases)")
    print("UCT formula with proper exploration-exploitation")
    print("Connect 4 environment with full game logic")
    print("MCTS finding optimal moves")
    print("Strong performance vs random baseline")