"""

Large Language Models as Commonsense Knowledge for Large-Scale Task Planning


LLM-MCTS planner (compact, research-oriented).
- Implements a Monte Carlo Tree Search where:
  * LLM acts as world model: given state + action -> next state description + immediate reward estimate
  * LLM acts as policy heuristic: given state -> ranked candidate actions
- Demonstration task: product-launch checklist planning

Caveats:
- This is a practical research implementation, simplified for clarity.
- You can choose LLM backend: "openai" (requires API key) or "hf" (local HF model).
"""

import os
import math
import random
import time
from copy import deepcopy
from typing import List, Tuple, Any, Optional
from dataclasses import dataclass, field

# choose backend; default to "openai" if key present; else "hf"
DEFAULT_BACKEND = "openai" if os.getenv("OPENAI_API_KEY") else "hf"

# choose model names
OPENAI_MODEL = "gpt-4o-mini"  # replace with desired model
HF_MODEL = "google/flan-t5-small"  # small, reasonably capable for testing

# LLM timeout / rate-limiting safe pause
LLM_DELAY = 0.6

# --------------------------
#  LLM Interface (OpenAI or HF)
# --------------------------
class LLMInterface:
    def __init__(self, backend: str = DEFAULT_BACKEND, hf_model: str = HF_MODEL, openai_model: str = OPENAI_MODEL, temp: float = 0.2):
        self.backend = backend
        self.temp = temp
        self.openai_model = openai_model
        self.hf_model = hf_model
        if backend == "openai":
            try:
                import openai
            except Exception as e:
                raise RuntimeError("openai package required for openai backend. pip install openai") from e
            self.openai = openai
        elif backend == "hf":
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
            except Exception as e:
                raise RuntimeError("transformers package required for hf backend. pip install transformers torch") from e
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model)
            self.pipeline = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        else:
            raise ValueError("backend must be 'openai' or 'hf'")

    def call(self, prompt: str, max_tokens: int = 200) -> str:
        """Single-turn LLM call returning text."""
        if self.backend == "openai":
            # Chat completion style
            # uses chat completions if supported; fallback to completion-like call
            time.sleep(LLM_DELAY)
            resp = self.openai.ChatCompletion.create(
                model=self.openai_model,
                messages=[{"role":"user","content":prompt}],
                temperature=self.temp,
                max_tokens=max_tokens,
            )
            # find assistant reply
            text = resp["choices"][0]["message"]["content"].strip()
            return text
        else:
            # HF pipeline call
            time.sleep(LLM_DELAY)
            out = self.pipeline(prompt, max_length=max_tokens, do_sample=True, temperature=self.temp)
            return out[0]["generated_text"].strip()

# --------------------------
#  Planning domain: product launch
# --------------------------
"""
State representation (simple text):
- 'state' is a bullet-list of completed items and current resources (budget, days_left).
Action space:
- templated actions like "Prepare marketing plan", "Book venue", "Design landing page", "Hire contractor", etc.
We will let the LLM suggest candidate actions (policy), and the LLM world-model will predict effects of performing an action.
"""

DEFAULT_ACTIONS = [
    "Define target customers and messaging",
    "Create product landing page",
    "Prepare marketing plan (email + social)",
    "Design product packaging",
    "Set pricing and offers",
    "Book launch venue or virtual platform",
    "Arrange a demo and QA session",
    "Prepare press release and outreach list",
    "Run an initial paid ad test",
    "Coordinate supply chain & inventory",
    "Arrange customer support channels",
    "Set up analytics and tracking",
    "Create launch day checklist",
]

@dataclass
class PlanState:
    completed: List[str] = field(default_factory=list)
    days_left: int = 30
    budget: float = 5000.0
    notes: List[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [
            f"Days left: {self.days_left}",
            f"Budget: ${self.budget:.2f}",
            "Completed items:"
        ]
        if self.completed:
            lines += [f"- {c}" for c in self.completed]
        else:
            lines += ["- (none)"]
        if self.notes:
            lines += ["Notes:"] + [f"- {n}" for n in self.notes]
        return "\n".join(lines)

    def clone(self):
        return deepcopy(self)

# --------------------------
#  Simple LLM-driven domain functions
# --------------------------
def build_world_model_prompt(state: PlanState, action: str) -> str:
    return (
        "You are a helpful planner engine that predicts the immediate outcome of taking an action in a product launch planning context.\n\n"
        f"Current state:\n{state.text()}\n\n"
        f"Action to simulate: {action}\n\n"
        "Respond with three lines only, labeled EXACTLY as follows:\n"
        "NEXT_STATE: <one-line summary of the new state (days_left and budget updated, and newly completed item)>\n"
        "REWARD: <numeric reward between -1.0 and +1.0 representing how good this action is right now>\n"
        "NOTE: <concise note about side-effects or important dependencies (one sentence)>"
    )

def build_policy_prompt(state: PlanState, candidate_pool: List[str], k: int = 6) -> str:
    # ask LLM to rank candidate actions and optionally add any new suggestions
    return (
        "You are an expert startup operations planner. Given the current state, rank the most promising next steps (max {} choices).\n\n"
        "State:\n"
        "{}\n\n"
        "Candidate actions (you can reorder and optionally add 1-2 additional short actions):\n- {}\n\n"
        "Return a numbered list (1..n) of your top picks with very brief justification (one short phrase per item)."
    ).format(k, state.text(), "\n- ".join(candidate_pool))

# parse world model reply
def parse_world_model_reply(reply: str) -> Tuple[PlanState, float, str]:
    # naive parsing, robust-ish
    lines = [l.strip() for l in reply.splitlines() if l.strip()]
    next_state_text = ""
    reward = 0.0
    note = ""
    for l in lines:
        if l.upper().startswith("NEXT_STATE:"):
            next_state_text = l.split(":",1)[1].strip()
        elif l.upper().startswith("REWARD:"):
            try:
                reward = float(l.split(":",1)[1].strip())
            except:
                # fallback: try extract float anywhere
                import re
                m = re.search(r"[-+]?\d*\.\d+|\d+", l)
                if m: reward = float(m.group(0))
        elif l.upper().startswith("NOTE:"):
            note = l.split(":",1)[1].strip()
    # very simple update logic: try to extract days/budget/completed from next_state_text heuristically
    ns = PlanState()
    ns.completed = []
    ns.notes = []
    ns.days_left = None
    ns.budget = None
    # parse days and budget
    import re
    d = re.search(r"Days? left[: ]*([0-9]+)", next_state_text, re.I)
    b = re.search(r"Budget[: ]*\$?([0-9]+(?:\.[0-9]+)?)", next_state_text, re.I)
    if d:
        ns.days_left = int(d.group(1))
    if b:
        ns.budget = float(b.group(1))
    # parse completed item
    comp = re.search(r"completed[: ]*(.+)$", next_state_text, re.I)
    if comp:
        ns.completed = [comp.group(1).strip()]
    # fallback: put raw next_state_text into notes if parsing failed
    if ns.days_left is None: ns.days_left = 0
    if ns.budget is None: ns.budget = 0.0
    if not ns.completed:
        ns.notes = [next_state_text]
    return ns, reward, note

# --------------------------
#  MCTS implementation
# --------------------------
class MCTSNode:
    def __init__(self, state: PlanState, parent=None, action_from_parent: Optional[str]=None):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children: List['MCTSNode'] = []
        self._untried_actions: Optional[List[str]] = None
        self.visits = 0
        self.value = 0.0  # cumulative value

    def untried_actions(self, candidate_pool: List[str]) -> List[str]:
        if self._untried_actions is None:
            # actions that are not yet completed and not in children
            completed = set(self.state.completed)
            all_actions = [a for a in candidate_pool if a not in completed]
            # remove those that we already expanded
            expanded = {c.action_from_parent for c in self.children if c.action_from_parent}
            self._untried_actions = [a for a in all_actions if a not in expanded]
        return self._untried_actions

    def uct_select_child(self, c_param: float = 1.4) -> 'MCTSNode':
        # UCT: value/visits + c * sqrt(2 ln N / n)
        choices = []
        for ch in self.children:
            if ch.visits == 0:
                score = float('inf')
            else:
                score = (ch.value / ch.visits) + c_param * math.sqrt(math.log(self.visits + 1) / ch.visits)
            choices.append((score, ch))
        # pick highest
        _, best = max(choices, key=lambda x: x[0])
        return best

    def add_child(self, action: str, next_state: PlanState):
        child = MCTSNode(state=next_state, parent=self, action_from_parent=action)
        self.children.append(child)
        # invalidate untried actions cache
        self._untried_actions = None
        return child

    def update(self, reward: float):
        self.visits += 1
        self.value += reward

class MCTS:
    def __init__(self, llm: LLMInterface, candidate_pool: List[str], iterations: int = 100, c_param: float = 1.2, rollout_depth: int = 4):
        self.llm = llm
        self.candidate_pool = candidate_pool
        self.iterations = iterations
        self.c_param = c_param
        self.rollout_depth = rollout_depth

    def search(self, root_state: PlanState) -> Tuple[List[Tuple[str,float]], MCTSNode]:
        root = MCTSNode(state=root_state)
        for it in range(self.iterations):
            node = root
            # 1. Selection
            while node.untried_actions(self.candidate_pool) == [] and node.children:
                node = node.uct_select_child(self.c_param)
            # 2. Expansion
            untried = node.untried_actions(self.candidate_pool)
            if untried:
                # use LLM policy to rank a small batch of actions; pick top
                policy_prompt = build_policy_prompt(node.state, untried[:max(6,len(untried))], k=6)
                ranking = self.llm.call(policy_prompt, max_tokens=300)
                # parse first suggested action if available
                first_action = None
                for line in ranking.splitlines():
                    line = line.strip()
                    if line and line[0].isdigit() and '.' in line:
                        try:
                            _, rest = line.split('.',1)
                            first_action = rest.strip().split('-')[0].strip()
                            break
                        except:
                            continue
                if not first_action:
                    first_action = random.choice(untried)
                # simulate world-model to get next_state and reward
                wprompt = build_world_model_prompt(node.state, first_action)
                wm_reply = self.llm.call(wprompt, max_tokens=200)
                next_state_partial, reward_est, note = parse_world_model_reply(wm_reply)
                # create next state by merging info
                next_state = node.state.clone()
                if next_state_partial.completed:
                    next_state.completed += next_state_partial.completed
                if next_state_partial.notes:
                    next_state.notes += next_state_partial.notes
                # heuristics: decrease days/budget lightly if not provided explicitly
                next_state.days_left = max(0, node.state.days_left - 2)
                next_state.budget = max(0.0, node.state.budget - min(1000.0, node.state.budget*0.05))
                child = node.add_child(first_action, next_state)
                node = child
                rollout_reward = self.simulate(node.state, depth=self.rollout_depth)
                # combine rollout reward and immediate LLM reward
                total_reward = 0.6 * reward_est + 0.4 * rollout_reward
            else:
                # no untried actions - leaf
                rollout_reward = self.simulate(node.state, depth=self.rollout_depth)
                total_reward = rollout_reward

            # 4. Backpropagate
            while node is not None:
                node.update(total_reward)
                node = node.parent
        # after iterations, rank root children by value/visits
        choices = []
        for ch in root.children:
            avg = (ch.value / ch.visits) if ch.visits>0 else 0.0
            choices.append((ch.action_from_parent, avg))
        choices.sort(key=lambda x: x[1], reverse=True)
        return choices, root

    def simulate(self, state: PlanState, depth: int = 3) -> float:
        # simple rollout: greedily ask LLM policy for next actions and world model to score them,
        # accumulating rewards; average reward returned.
        total = 0.0
        s = state.clone()
        for d in range(depth):
            pool = [a for a in self.candidate_pool if a not in s.completed]
            if not pool:
                break
            prompt = build_policy_prompt(s, pool[:8], k=4)
            policy_text = self.llm.call(prompt, max_tokens=200)
            # pick first suggested action
            chosen = None
            for line in policy_text.splitlines():
                line = line.strip()
                if line and line[0].isdigit() and '.' in line:
                    try:
                        _, rest = line.split('.',1)
                        chosen = rest.strip().split('-')[0].strip()
                        break
                    except:
                        continue
            if not chosen:
                chosen = random.choice(pool)
            wprompt = build_world_model_prompt(s, chosen)
            wm_reply = self.llm.call(wprompt, max_tokens=200)
            nxt_partial, r, note = parse_world_model_reply(wm_reply)
            total += r
            if nxt_partial.completed:
                s.completed += nxt_partial.completed
            s.days_left = max(0, s.days_left - 2)
            s.budget = max(0.0, s.budget - min(800.0, s.budget * 0.04))
        avg = total / max(1, depth)
        return avg

# --------------------------
#  Example run
# --------------------------
def demo_run(backend: str = DEFAULT_BACKEND, iterations: int = 60):
    print(f"Using backend: {backend}")
    llm = LLMInterface(backend=backend)
    # start state for product launch
    start = PlanState(completed=[], days_left=30, budget=5000.0, notes=["Launch MVP in 30 days", "Small marketing budget"])
    print("Start state:\n", start.text(), "\n")
    mcts = MCTS(llm, candidate_pool=DEFAULT_ACTIONS, iterations=iterations, c_param=1.2, rollout_depth=3)
    start_time = time.time()
    choices, root = mcts.search(start)
    dur = time.time() - start_time
    print(f"\nMCTS completed {iterations} iterations in {dur:.1f}s. Top suggestions (action, score):")
    for a, s in choices[:8]:
        print(f" - {a}  (avg score {s:.3f})")
    # print best path (greedy by child avg)
    if choices:
        best_action = choices[0][0]
        print("\nBest first action recommended:", best_action)
        # print rough plan by following top children greedily (depth-limited)
        node = None
        for ch in root.children:
            if ch.action_from_parent == best_action:
                node = ch; break
        plan = []
        cur = node
        if cur:
            plan.append((cur.action_from_parent, cur.state.text()))
            for _ in range(4):
                if not cur.children:
                    break
                # pick child with max avg
                cur = max(cur.children, key=lambda c: (c.value / c.visits) if c.visits>0 else 0.0)
                plan.append((cur.action_from_parent, cur.state.text()))
        print("\nGreedy plan (approx):")
        for i,(a,s) in enumerate(plan):
            print(f"{i+1}. {a}\n   state snapshot: {s}\n")

if __name__ == "__main__":
    # run demo
    demo_run(backend=DEFAULT_BACKEND, iterations=60)
