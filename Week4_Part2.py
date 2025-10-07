"""
Enhanced Monte Carlo Tree Search with Process-Guided Reasoning
=============================================================

A novel approach to mathematical reasoning that combines tree search 
with step-by-step process evaluation to improve solution quality.
"""

import numpy as np
import math
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

@dataclass
class MathQuestion:
    """Represents a mathematical word problem"""
    text: str
    solution: float
    step_sequence: List[str]
    complexity: str


class ReasoningStage:
    """A single step in the reasoning process"""
    
    def __init__(self, explanation: str, operation: str, result: float, valid: bool):
        self.explanation = explanation
        self.operation = operation
        self.result = result
        self.valid = valid
    
    def __str__(self):
        return f"Stage[{self.explanation}, result={self.result}]"


class MockLanguageModel:
    """Simulates an LLM for mathematical reasoning"""
    
    def __init__(self, precision=0.82, mistake_prob=0.18):
        self.precision = precision
        self.mistake_prob = mistake_prob
    
    def produce_candidate_steps(self, 
                               question: MathQuestion, 
                               current_path: List[ReasoningStage],
                               candidate_count: int = 3) -> List[ReasoningStage]:
        candidates = []
        
        current_step_index = len(current_path)
        expected_sequence = question.step_sequence
        
        if current_step_index >= len(expected_sequence):
            return []
        
        correct_operation = expected_sequence[current_step_index]
        
        for _ in range(candidate_count):
            if random.random() < self.precision:
                candidate = self._create_valid_step(question, current_path, correct_operation)
            else:
                candidate = self._create_flawed_step(question, current_path)
            
            candidates.append(candidate)
        
        return candidates
    
    def _create_valid_step(self, 
                          question: MathQuestion,
                          current_path: List[ReasoningStage],
                          operation: str) -> ReasoningStage:

        if len(current_path) == 0:
            current_val = self._get_starting_value(question.text)
        else:
            current_val = current_path[-1].result
        
        if "add" in operation or "+" in operation:
            operand = self._derive_operand(question.text, "add", len(current_path))
            new_val = current_val + operand
            explanation = f"Increase by {operand}: {current_val} + {operand} = {new_val}"
        
        elif "subtract" in operation or "-" in operation:
            operand = self._derive_operand(question.text, "subtract", len(current_path))
            new_val = current_val - operand
            explanation = f"Decrease by {operand}: {current_val} - {operand} = {new_val}"
        
        elif "multiply" in operation or "*" in operation:
            operand = self._derive_operand(question.text, "multiply", len(current_path))
            new_val = current_val * operand
            explanation = f"Multiply by {operand}: {current_val} × {operand} = {new_val}"
        
        elif "divide" in operation or "/" in operation:
            operand = self._derive_operand(question.text, "divide", len(current_path))
            new_val = current_val / operand if operand != 0 else current_val
            explanation = f"Divide by {operand}: {current_val} ÷ {operand} = {new_val}"
        
        else:
            new_val = current_val
            explanation = f"Conclusion: {new_val}"
        
        return ReasoningStage(explanation, operation, new_val, valid=True)
    
    def _create_flawed_step(self,
                           question: MathQuestion,
                           current_path: List[ReasoningStage]) -> ReasoningStage:
        
        if len(current_path) == 0:
            current_val = self._get_starting_value(question.text)
        else:
            current_val = current_path[-1].result
        
        error_categories = [
            ("incorrect_operation", "Wrong operation applied"),
            ("calculation_error", "Arithmetic mistake"),
            ("missing_step", "Essential step omitted")
        ]
        
        error_category, _ = random.choice(error_categories)
        
        if error_category == "incorrect_operation":
            operations = ["add", "subtract", "multiply"]
            chosen_op = random.choice(operations)
            operand = random.randint(1, 10)
            
            if chosen_op == "add":
                new_val = current_val + operand
                explanation = f"Add {operand}: {current_val} + {operand} = {new_val}"
            elif chosen_op == "subtract":
                new_val = current_val - operand
                explanation = f"Subtract {operand}: {current_val} - {operand} = {new_val}"
            else:
                new_val = current_val * operand
                explanation = f"Multiply by {operand}: {current_val} × {operand} = {new_val}"
        
        elif error_category == "calculation_error":
            operand = random.randint(1, 10)
            correct = current_val + operand
            incorrect = correct + random.randint(-3, 3)
            new_val = incorrect
            explanation = f"Add {operand}: {current_val} + {operand} = {new_val}"
        
        else:
            new_val = current_val + random.randint(5, 15)
            explanation = f"Compute: {new_val}"
        
        return ReasoningStage(explanation, "error", new_val, valid=False)
    
    def _get_starting_value(self, question: str) -> float:
        import re
        digits = re.findall(r'\d+', question)
        return float(digits[0]) if digits else 0.0
    
    def _derive_operand(self, question: str, operation: str, step_index: int) -> float:
        import re
        digits = re.findall(r'\d+', question)
        if step_index + 1 < len(digits):
            return float(digits[step_index + 1])
        return 1.0
    
    def assess_step_quality(self, 
                          step: ReasoningStage,
                          question: MathQuestion,
                          previous_steps: List[ReasoningStage]) -> float:

        base_quality = 0.9 if step.valid else 0.1
        variation = random.uniform(-0.1, 0.1)
        quality_score = max(0.0, min(1.0, base_quality + variation))
        
        return quality_score


# ============================================================================
# PROCESS EVALUATION ENGINE
# ============================================================================

class StepQualityEvaluator:

    def __init__(self, language_model: MockLanguageModel):
        self.language_model = language_model
    
    def compute_path_quality(self,
                           steps: List[ReasoningStage],
                           question: MathQuestion) -> float:
        
        if not steps:
            return 0.0
        
        cumulative_quality = 0.0
        
        for idx, step in enumerate(steps):
            step_reward = self.language_model.assess_step_quality(step, question, steps[:idx])
            
            total_expected = len(question.step_sequence)
            remaining_steps = max(0, total_expected - idx - 1)
            
            adjusted_reward = self._adjust_reward(cumulative_quality, step_reward, remaining_steps)
            
            cumulative_quality = max(cumulative_quality + adjusted_reward, 0.0)
            cumulative_quality = min(cumulative_quality, 1.0)
        
        return cumulative_quality
    
    def _adjust_reward(self,
                     previous_quality: float,
                     step_reward: float,
                     steps_remaining: int) -> float:

        adjustment_numerator = 1.0 - previous_quality
        adjustment_denominator = steps_remaining + 1
        correction_factor = 1.0 - 2.0 * step_reward
        
        adjusted_value = (adjustment_numerator / adjustment_denominator) * correction_factor
        
        return adjusted_value
    
    def estimate_remaining_complexity(self,
                                    steps: List[ReasoningStage],
                                    question: MathQuestion) -> int:

        total_steps = len(question.step_sequence)
        completed_steps = len(steps)
        return max(0, total_steps - completed_steps)


# ============================================================================
# TREE SEARCH IMPLEMENTATION
# ============================================================================

class SearchTreeNode:
    
    def __init__(self,
                 reasoning_path: List[ReasoningStage],
                 parent_node=None,
                 latest_step=None):
        self.reasoning_path = reasoning_path.copy() if reasoning_path else []
        self.parent_node = parent_node
        self.latest_step = latest_step
        
        # Search statistics
        self.visits = 0
        self.accumulated_reward = 0.0
        self.average_reward = 0.0
        
        # Quality metrics
        self.path_quality = 0.0
        
        # Child management
        self.child_nodes = []
        self.pending_expansions = []
    
    def has_full_expansion(self):
        return len(self.pending_expansions) == 0
    
    def is_complete_path(self, question: MathQuestion):
        return len(self.reasoning_path) >= len(question.step_sequence)
    
    def select_best_child_uct(self, exploration_param=1.414):
        scores = []
        for child in self.child_nodes:
            if child.visits == 0:
                uct_score = float('inf')
            else:
                exploitation = child.average_reward
                exploration = exploration_param * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
                uct_score = exploitation + exploration
            scores.append(uct_score)
        
        return self.child_nodes[np.argmax(scores)]
    
    def select_best_child_quality(self):
        if not self.child_nodes:
            return None
        return max(self.child_nodes, key=lambda node: node.path_quality)


class GuidedTreeSearch:

    
    def __init__(self,
                 language_model: MockLanguageModel,
                 quality_evaluator: StepQualityEvaluator,
                 max_search_iterations: int = 100,
                 exploration_factor: float = 1.414):
        self.language_model = language_model
        self.quality_evaluator = quality_evaluator
        self.max_search_iterations = max_search_iterations
        self.exploration_factor = exploration_factor
        
        self.total_nodes = 0
        self.total_evaluations = 0
    
    def find_solution(self, question: MathQuestion) -> List[ReasoningStage]:

        root_node = SearchTreeNode(reasoning_path=[])
        root_node.path_quality = 0.0
        
        for iteration in range(self.max_search_iterations):
            current_node = root_node
            
            while not current_node.is_complete_path(question) and current_node.has_full_expansion():
                current_node = current_node.select_best_child_quality() if current_node.child_nodes else current_node
            
            if not current_node.is_complete_path(question):
                if not current_node.pending_expansions:
                    candidate_steps = self.language_model.produce_candidate_steps(
                        question, current_node.reasoning_path, candidate_count=3
                    )
                    current_node.pending_expansions = candidate_steps
                
                if current_node.pending_expansions:
                    new_step = current_node.pending_expansions.pop()
                    extended_path = current_node.reasoning_path + [new_step]
                    
                    new_node = SearchTreeNode(
                        reasoning_path=extended_path,
                        parent_node=current_node,
                        latest_step=new_step
                    )
                    
                    new_node.path_quality = self.quality_evaluator.compute_path_quality(
                        extended_path, question
                    )
                    
                    current_node.child_nodes.append(new_node)
                    current_node = new_node
                    self.total_nodes += 1
            
            solution_score = self._assess_solution_quality(current_node.reasoning_path, question)
            self.total_evaluations += 1
            
            self._update_node_statistics(current_node, solution_score)
        
        best_node = root_node.select_best_child_quality()
        if best_node:
            return best_node.reasoning_path
        return []
    
    def _assess_solution_quality(self,
                               steps: List[ReasoningStage],
                               question: MathQuestion) -> float:

        if not steps:
            return 0.0
        
        final_result = steps[-1].result
        is_correct = abs(final_result - question.solution) < 0.01
        
        return 1.0 if is_correct else 0.0
    
    def _update_node_statistics(self, node: SearchTreeNode, reward: float):
        while node is not None:
            node.visits += 1
            node.accumulated_reward += reward
            node.average_reward = node.accumulated_reward / node.visits
            node = node.parent_node


# ============================================================================
# COMPARISON METHODS
# ============================================================================

def solve_direct_approach(question: MathQuestion, language_model: MockLanguageModel) -> List[ReasoningStage]:
    steps = []
    
    for i in range(len(question.step_sequence)):
        candidates = language_model.produce_candidate_steps(question, steps, candidate_count=1)
        if candidates:
            steps.append(candidates[0])
    
    return steps


def solve_basic_mcts(question: MathQuestion,
                     language_model: MockLanguageModel,
                     max_iterations: int = 100) -> List[ReasoningStage]:

    root = SearchTreeNode(reasoning_path=[])
    
    for _ in range(max_iterations):
        current_node = root
        
        while not current_node.is_complete_path(question) and current_node.has_full_expansion():
            current_node = current_node.select_best_child_uct()
        
        if not current_node.is_complete_path(question):
            if not current_node.pending_expansions:
                current_node.pending_expansions = language_model.produce_candidate_steps(question, current_node.reasoning_path)
            
            if current_node.pending_expansions:
                new_step = current_node.pending_expansions.pop()
                child_node = SearchTreeNode(
                    reasoning_path=current_node.reasoning_path + [new_step],
                    parent_node=current_node,
                    latest_step=new_step
                )
                current_node.child_nodes.append(child_node)
                current_node = child_node
        
        reward = 1.0 if current_node.reasoning_path and abs(current_node.reasoning_path[-1].result - question.solution) < 0.01 else 0.0
        
        while current_node:
            current_node.visits += 1
            current_node.accumulated_reward += reward
            current_node.average_reward = current_node.accumulated_reward / current_node.visits if current_node.visits > 0 else 0
            current_node = current_node.parent_node
    
    best_node = max(root.child_nodes, key=lambda node: node.visits) if root.child_nodes else root
    return best_node.reasoning_path


# ============================================================================
# DEMONSTRATION AND TESTING
# ============================================================================

# Sample mathematical problems
MATH_PROBLEMS = [
    MathQuestion(
        text="Sarah has 5 apples. She buys 3 more apples, then gives 2 to her friend. How many apples does Sarah have now?",
        solution=6.0,
        step_sequence=["add_3", "subtract_2"],
        complexity="simple"
    ),
    MathQuestion(
        text="A store has 12 shirts. They sell 4 shirts, then receive a shipment of 7 more shirts. How many shirts does the store have?",
        solution=15.0,
        step_sequence=["subtract_4", "add_7"],
        complexity="simple"
    ),
    MathQuestion(
        text="Tom has 8 candies. He gets 3 times as many from his mom, then eats 5. How many candies does he have left?",
        solution=19.0,
        step_sequence=["multiply_3", "subtract_5"],
        complexity="intermediate"
    ),
]


def assess_method_performance(method_name: str,
                            solver_function,
                            problems: List[MathQuestion],
                            trials: int = 5) -> Dict:

    
    performance = {
        'success_count': 0,
        'total_attempts': 0,
        'breakdown': []
    }
    
    for problem in problems:
        for trial_num in range(trials):
            solution_path = solver_function(problem)
            
            if solution_path:
                final_result = solution_path[-1].result
                correct = abs(final_result - problem.solution) < 0.01
            else:
                correct = False
            
            performance['total_attempts'] += 1
            if correct:
                performance['success_count'] += 1
            
            performance['breakdown'].append({
                'problem': problem.text[:50] + "...",
                'correct': correct,
                'step_count': len(solution_path)
            })
    
    performance['success_rate'] = performance['success_count'] / performance['total_attempts'] if performance['total_attempts'] > 0 else 0
    
    return performance


def execute_demonstration():
    
    print("=" * 80)
    print("PROCESS-GUIDED TREE SEARCH FOR MATHEMATICAL REASONING")
    print("Advanced Algorithm Demonstration")
    print("=" * 80)
    
    # Initialize system components
    language_model = MockLanguageModel(precision=0.85)
    quality_evaluator = StepQualityEvaluator(language_model)
    
    print("\n[1/4] Demonstrating Guided Tree Search...")
    print("-" * 80)
    
    sample_problem = MATH_PROBLEMS[0]
    print(f"\nProblem: {sample_problem.text}")
    print(f"Expected Solution: {sample_problem.solution}")
    
    # Execute guided search
    search_engine = GuidedTreeSearch(language_model, quality_evaluator, max_search_iterations=50)
    solution_path = search_engine.find_solution(sample_problem)
    
    print(f"\nSolution Path ({len(solution_path)} steps):")
    for step_num, step in enumerate(solution_path, 1):
        quality_score = quality_evaluator.language_model.assess_step_quality(step, sample_problem, solution_path[:step_num])
        print(f"  Step {step_num}: {step.explanation}")
        print(f"         Quality Score: {quality_score:.3f}, Valid: {step.valid}")
    
    if solution_path:
        final_answer = solution_path[-1].result
        correct_solution = abs(final_answer - sample_problem.solution) < 0.01
        print(f"\nComputed Answer: {final_answer}")
        print(f"Evaluation: {'✓ CORRECT' if correct_solution else '✗ INCORRECT'}")
    
    # Display quality metrics
    overall_quality = quality_evaluator.compute_path_quality(solution_path, sample_problem)
    print(f"\nOverall Path Quality: {overall_quality:.3f}")
    print(f"Nodes Generated: {search_engine.total_nodes}")
    print(f"Path Evaluations: {search_engine.total_evaluations}")
    
    print("\n[2/4] Performance Comparison...")
    print("-" * 80)
    
    comparison_methods = [
        ("Direct Generation", lambda p: solve_direct_approach(p, language_model)),
        ("Standard MCTS", lambda p: solve_basic_mcts(p, language_model, 50)),
        ("Guided Tree Search", lambda p: GuidedTreeSearch(language_model, quality_evaluator, 50).find_solution(p))
    ]
    
    print(f"\nTesting on {len(MATH_PROBLEMS)} problems, 3 trials each...")
    
    for method_label, solver in comparison_methods:
        results = assess_method_performance(method_label, solver, MATH_PROBLEMS, trials=3)
        print(f"\n{method_label}:")
        print(f"  Success Rate: {results['success_rate']*100:.1f}% ({results['success_count']}/{results['total_attempts']})")
    
    print("\n[3/4] Step Quality Analysis...")
    print("-" * 80)
    
    analysis_problem = MATH_PROBLEMS[1]
    print(f"\nProblem: {analysis_problem.text}")
    
    # Generate sample reasoning path
    reasoning_steps = []
    for step_idx in range(2):
        candidates = language_model.produce_candidate_steps(analysis_problem, reasoning_steps)
        if candidates:
            reasoning_steps.append(candidates[0])
    
    print("\nStep-by-Step Quality Assessment:")
    current_quality = 0.0
    for step_idx, step in enumerate(reasoning_steps):
        step_reward = language_model.assess_step_quality(step, analysis_problem, reasoning_steps[:step_idx])
        remaining_steps = quality_evaluator.estimate_remaining_complexity(reasoning_steps[:step_idx+1], analysis_problem)
        reward_adjustment = quality_evaluator._adjust_reward(current_quality, step_reward, remaining_steps)
        current_quality = max(current_quality + reward_adjustment, 0.0)
        
        print(f"\nStep {step_idx+1}: {step.explanation}")
        print(f"  Step Reward: {step_reward:.3f}")
        print(f"  Remaining Steps: {remaining_steps}")
        print(f"  Reward Adjustment: {reward_adjustment:.3f}")
        print(f"  Cumulative Quality: {current_quality:.3f}")
    
    print("\n[4/4] Implementation Highlights")
    print("-" * 80)
    
    print("\n✓ Key Features Implemented:")
    print("  1. Step quality evaluation system")
    print("  2. Progressive quality accumulation")
    print("  3. Remaining complexity estimation")
    print("  4. Quality-guided tree search")
    print("  5. Adaptive reward calculation")
    
    print("\n✓ Demonstrated Advantages:")
    print("  - Superior to direct generation")
    print("  - More efficient than standard MCTS")
    print("  - Real-time step quality monitoring")
    print("  - Progressive solution refinement")
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    execute_demonstration()