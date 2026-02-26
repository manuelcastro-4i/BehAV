#!/usr/bin/env python3
"""
BehAV Director - Orchestration node that connects instruction decomposition to the planner.

Accepts a natural language navigation instruction, decomposes it via GPT-4, builds
CLIPSeg prompts + cost values, and publishes the configuration as JSON to /behav_config.

Goal extraction: GPT-4 also parses the destination and distance from the instruction.
  "go to the red marker at 17 metres" → goal_radius=17.0  (explicit distance)
  "reach the end of the path"         → goal_radius=BEHAV_GOAL_RADIUS (fallback, no distance given)

Usage:
    python3 behav-director.py --instruction "Stay on the pavement, avoid grass, go to the red marker at 17 m"
    python3 behav-director.py --demo                     # hardcoded demo config, no API key needed
    python3 behav-director.py --interactive              # prompt for instruction at runtime

Environment:
    OPENAI_API_KEY  - Required for GPT-4 decomposition (skipped in demo mode)
    ROS_DOMAIN_ID   - DDS domain (default 0)
"""

import argparse
import importlib.util
import json
import os
import sys

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Works both in-repo (planning/ next to instruction-decomposition/) and in container
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DECOMP_CANDIDATES = [
    os.path.join(_SCRIPT_DIR, '..', 'instruction-decomposition', 'GPT4o',
                 'instruction-to-behavioral-costs.py'),
    '/app/instruction-decomposition/GPT4o/instruction-to-behavioral-costs.py',
]

# ---------------------------------------------------------------------------
# Action → CLIPSeg cost mapping
# The optimizer MINIMIZES total_cost.  expected_behav += max_behav_cost/255
# is ADDED to total_cost, so:
#   HIGH cost_value → HIGH pixel values → HIGH total_cost → region is PENALIZED (avoided)
#   LOW  cost_value → LOW  pixel values → LOW  total_cost → region is PREFERRED
# ---------------------------------------------------------------------------
_ACTION_COST_MAP = [
    ('stay on',     0.1),   # LOW  = preferred, robot stays here
    ('follow',      0.1),
    ('stay away',   0.9),   # HIGH = penalized, robot avoids
    ('do not go',   0.9),
    ('avoid',       0.9),
    ('yield',       0.6),
    ('slow down',   0.5),
    ('stop for',    1.0),
    ('stop',        1.0),
]

# Fallback config used when no OpenAI key is present or --demo is specified
# grass/vegetation = 0.9 → HIGH penalty → robot avoids going off-path
# pavement         = 0.1 → LOW  penalty → robot prefers staying on road
DEMO_CONFIG = {
    'prompts': ['pavement', 'green grass', 'vegetation'],
    'costs':   [0.1, 0.9, 0.9],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action_to_cost(action: str) -> float:
    """Map a behavioral action string to a CLIPSeg cost value (0–1)."""
    lower = action.lower().strip()
    for keyword, cost in _ACTION_COST_MAP:
        if keyword in lower:
            return cost
    return 0.5  # neutral / unknown


def _load_decomp_module():
    """Load the instruction-to-behavioral-costs module by file path."""
    for candidate in _DECOMP_CANDIDATES:
        path = os.path.abspath(candidate)
        if os.path.exists(path):
            spec = importlib.util.spec_from_file_location('_decomp', path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Could not locate instruction-to-behavioral-costs.py. "
        f"Searched: {_DECOMP_CANDIDATES}"
    )


def decompose_instruction(instruction: str) -> dict:
    """
    Call GPT-4 to decompose *instruction* and return a dict with keys
    'prompts' (list[str]) and 'costs' (list[float]).
    """
    mod = _load_decomp_module()
    client = mod.get_openai_client()

    print(f"[Director] Calling GPT-4 to decompose: {instruction!r}")
    breakdown = mod.get_instruction_breakdown(client, instruction)
    print(f"[Director] Breakdown: {breakdown}")

    behavioral_actions = breakdown.get('behavioral_actions', [])
    behavioral_targets = breakdown.get('behavioral_targets', [])

    if not behavioral_targets:
        print("[Director] No behavioral targets found — using demo CLIPSeg config")
        return DEMO_CONFIG.copy()

    prompts, costs = [], []
    n_actions = len(behavioral_actions)
    for i, target in enumerate(behavioral_targets):
        action = behavioral_actions[i] if i < n_actions else (
            behavioral_actions[0] if n_actions else ''
        )
        cost = _action_to_cost(action)
        print(f"  action={action!r:20s}  target={target!r:20s}  → cost={cost}")
        prompts.append(target)
        costs.append(cost)

    return {'prompts': prompts, 'costs': costs}


def extract_goal_from_instruction(instruction: str, fallback_radius: float) -> tuple:
    """
    Ask GPT-4 to extract the destination and distance from *instruction*.

    Returns (goal_radius, goal_description):
      - goal_radius      : extracted metres, or fallback_radius if not stated
      - goal_description : human-readable string, or None
    """
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        return fallback_radius, None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"[Director] Goal extraction: could not init OpenAI client ({e})")
        return fallback_radius, None

    prompt = (
        f'From this robot navigation instruction extract the destination/goal:\n\n'
        f'"{instruction}"\n\n'
        f'Return ONLY valid JSON with two keys:\n'
        f'  "goal_description": the destination landmark or place (string, or null if none)\n'
        f'  "goal_distance_meters": distance to goal in metres (number, or null if not stated)\n\n'
        f'Examples:\n'
        f'  "go to the red marker at 17 m"   → {{"goal_description":"red marker","goal_distance_meters":17.0}}\n'
        f'  "reach the building 50 m ahead"  → {{"goal_description":"building","goal_distance_meters":50.0}}\n'
        f'  "go to the end of the path"      → {{"goal_description":"end of path","goal_distance_meters":null}}\n'
        f'  "stay on the pavement"           → {{"goal_description":null,"goal_distance_meters":null}}\n'
        f'\nReturn ONLY the JSON object, no other text.'
    )

    try:
        response = client.chat.completions.create(
            model='gpt-4',
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
        )
        raw = response.choices[0].message.content.strip()
        data = json.loads(raw)
        description = data.get('goal_description')
        distance    = data.get('goal_distance_meters')

        if isinstance(distance, (int, float)) and float(distance) > 0:
            radius = float(distance)
            print(f"[Director] Goal extracted: '{description}' at {radius} m")
            return radius, description
        elif description:
            print(f"[Director] Goal: '{description}' (no distance stated — fallback {fallback_radius} m)")
            return fallback_radius, description
        else:
            print(f"[Director] No destination in instruction — using fallback {fallback_radius} m")
            return fallback_radius, None

    except Exception as e:
        print(f"[Director] Goal extraction failed ({e}) — using fallback {fallback_radius} m")
        return fallback_radius, None


def build_config(behavioural_dict: dict,
                 goal_radius: float,
                 goal_theta: float,
                 goal_delta: float,
                 goal_description: str = None) -> dict:
    """Merge CLIPSeg config with goal parameters."""
    cfg = dict(behavioural_dict)
    cfg['goal_radius'] = goal_radius
    cfg['goal_theta']  = goal_theta
    cfg['goal_delta']  = goal_delta
    if goal_description:
        cfg['goal_description'] = goal_description
    return cfg


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class BehavDirector(Node):
    """Publishes /behav_config and re-publishes periodically."""

    def __init__(self, config: dict, republish_interval: float):
        super().__init__('behav_director')

        self._config = config
        self._republish_interval = republish_interval

        self._pub = self.create_publisher(String, '/behav_config', 10)

        # First publish after 1 s (DDS peer discovery), then periodically
        self._startup_timer = self.create_timer(1.0, self._first_publish)
        self._periodic_timer = None

        self.get_logger().info(
            f"BehavDirector ready | prompts={config.get('prompts')} | "
            f"goal_r={config.get('goal_radius')}m | "
            f"republish every {republish_interval}s"
        )

    def _first_publish(self):
        self._startup_timer.cancel()
        self._publish()
        self._periodic_timer = self.create_timer(
            self._republish_interval, self._publish
        )

    def _publish(self):
        msg = String()
        msg.data = json.dumps(self._config)
        self._pub.publish(msg)
        self.get_logger().info(
            f"Published /behav_config — prompts={self._config.get('prompts')} "
            f"goal_r={self._config.get('goal_radius')}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='BehAV Director: decompose an instruction and drive the planner.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --instruction "Stay on the pavement, avoid grass" --goal-radius 5.0
  %(prog)s --demo                   # no API key needed
  %(prog)s --interactive            # prompt for instruction at runtime
        """
    )
    parser.add_argument(
        '--instruction', '-I',
        help='Natural language navigation instruction'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Prompt for instruction interactively'
    )
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Use hardcoded demo config (no OpenAI key required)'
    )
    parser.add_argument(
        '--goal-radius', type=float, default=5.0, metavar='METERS',
        help='Goal distance from robot start position (default: 5.0)'
    )
    parser.add_argument(
        '--goal-theta', type=float, default=0.0, metavar='DEGREES',
        help='Goal heading angle in degrees (default: 0.0)'
    )
    parser.add_argument(
        '--goal-delta', type=float, default=0.0, metavar='DEGREES',
        help='Goal pose angle in degrees (default: 0.0)'
    )
    parser.add_argument(
        '--republish-interval', type=float, default=10.0, metavar='SECONDS',
        help='Re-publish interval in seconds (default: 10.0)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Resolve instruction and build behavioural config
    # ------------------------------------------------------------------
    instruction_text = None   # track for goal extraction

    if args.demo:
        print("[Director] Demo mode — using hardcoded CLIPSeg config")
        behav_dict = DEMO_CONFIG.copy()

    elif args.interactive:
        print("[Director] Interactive mode")
        instruction_text = input("Enter navigation instruction: ").strip()
        if not instruction_text:
            print("[Director] Empty instruction — falling back to demo config")
            behav_dict = DEMO_CONFIG.copy()
            instruction_text = None
        else:
            has_key = bool(os.environ.get('OPENAI_API_KEY'))
            if not has_key:
                print("[Director] OPENAI_API_KEY not set — using demo config")
                behav_dict = DEMO_CONFIG.copy()
            else:
                behav_dict = decompose_instruction(instruction_text)

    elif args.instruction:
        instruction_text = args.instruction
        has_key = bool(os.environ.get('OPENAI_API_KEY'))
        if not has_key:
            print("[Director] OPENAI_API_KEY not set — using demo config")
            behav_dict = DEMO_CONFIG.copy()
            instruction_text = None
        else:
            behav_dict = decompose_instruction(instruction_text)

    else:
        print("[Director] No instruction provided — using demo config. "
              "Pass --instruction, --demo, or --interactive.")
        behav_dict = DEMO_CONFIG.copy()

    # ------------------------------------------------------------------
    # Extract goal destination from instruction (if GPT-4 is available)
    # Falls back to --goal-radius / BEHAV_GOAL_RADIUS if no distance found
    # ------------------------------------------------------------------
    goal_radius      = args.goal_radius
    goal_description = None
    if instruction_text and bool(os.environ.get('OPENAI_API_KEY')):
        goal_radius, goal_description = extract_goal_from_instruction(
            instruction_text, fallback_radius=args.goal_radius
        )

    config = build_config(
        behav_dict, goal_radius, args.goal_theta, args.goal_delta, goal_description
    )
    print(f"[Director] Final config: {json.dumps(config, indent=2)}")

    # ------------------------------------------------------------------
    # Start ROS2 node
    # ------------------------------------------------------------------
    rclpy.init()
    node = BehavDirector(config, republish_interval=args.republish_interval)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
