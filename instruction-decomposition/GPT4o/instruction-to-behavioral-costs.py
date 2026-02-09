#!/usr/bin/env python3
"""
Instruction to Behavioral Costs - BehAV Navigation System

Breaks down natural language navigation instructions into:
- Landmarks (e.g., a building, stop sign)
- Navigation actions (e.g., go forward, turn left)
- Behavioral actions (e.g., stay on, avoid)
- Behavioral targets (e.g., pavement, grass)
- Behavioral costs for motion planning

Usage:
    # Interactive mode - prompts for instructions
    python instruction-to-behavioral-costs.py --interactive
    python instruction-to-behavioral-costs.py -i

    # Automatic mode - pass instruction as argument
    python instruction-to-behavioral-costs.py "Go forward until you see the red building"
    python instruction-to-behavioral-costs.py --instruction "Stay on the sidewalk, avoid grass"

    # Demo mode - runs with example instruction
    python instruction-to-behavioral-costs.py --demo

Docker:
    # Interactive mode
    docker-compose --profile test run --rm instruction_decomposition \\
        python /app/instruction-decomposition/GPT4o/instruction-to-behavioral-costs.py -i

    # With custom instruction
    docker-compose --profile test run --rm instruction_decomposition \\
        python /app/instruction-decomposition/GPT4o/instruction-to-behavioral-costs.py \\
        "Turn right at the tree, stay on pavement"

Environment:
    OPENAI_API_KEY - Required. Your OpenAI API key for GPT-4.
"""

from openai import OpenAI
import numpy as np
import ast
import os
import argparse
import sys


# Reference actions and their associated costs for behavioral planning
REFERENCE_ACTIONS = ['Stay on', 'Avoid', 'Yield', 'Stop']
REFERENCE_COSTS = [0, 0.5, 0.7, 1]

# Global verbose flag
VERBOSE = False

# Example instruction for demo mode
DEMO_INSTRUCTION = (
    'Go forward until you see a stop sign, then turn left and go straight '
    'until you see a white building, stay on the pavements, stop for red '
    'traffic lights, stay away from grass'
)


def get_openai_client():
    """Get OpenAI client using API key from environment variable."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set.\n"
            "Please set it with: export OPENAI_API_KEY=your-key-here"
        )
    return OpenAI(api_key=api_key)


def get_instruction_breakdown(client, language_instruction):
    """Parse instruction into landmarks, navigation actions, behavioral actions, and targets."""
    prompt = f"""Analyze this robot navigation instruction and extract components:

"{language_instruction}"

Extract these 4 categories:
- landmarks: physical objects used as reference points (fountain, building, tree, sign, etc.)
- navigation_actions: movement commands (turn right, go forward, follow, etc.)
- behavioral_actions: how to interact with surfaces/objects (avoid, stay on, follow, stop for, etc.)
- behavioral_targets: surfaces/objects for behavioral actions (path, puddles, grass, pavement, etc.)

Return ONLY a Python dictionary in this exact format, no explanation:
{{"landmarks": ["item1", "item2"], "navigation_actions": ["action1"], "behavioral_actions": ["action1"], "behavioral_targets": ["target1"]}}

Use empty lists [] if a category has no items."""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    instruction_breakdown_str = response.choices[0].message.content.strip()

    if VERBOSE:
        print(f"\n  [VERBOSE] Raw GPT-4 response:")
        print(f"  {instruction_breakdown_str}\n")

    # Try to extract JSON/dict from the response
    try:
        instruction_breakdown_dict = ast.literal_eval(instruction_breakdown_str)
    except (ValueError, SyntaxError):
        # Try to find a dict pattern in the response
        import re
        match = re.search(r'\{[^{}]*\}', instruction_breakdown_str, re.DOTALL)
        if match:
            try:
                instruction_breakdown_dict = ast.literal_eval(match.group())
            except (ValueError, SyntaxError):
                print(f"  Warning: Could not parse GPT-4 response. Raw output:")
                print(f"  {instruction_breakdown_str[:500]}")
                return {}
        else:
            print(f"  Warning: Could not parse GPT-4 response. Raw output:")
            print(f"  {instruction_breakdown_str[:500]}")
            return {}

    return instruction_breakdown_dict


def get_list_by_key(dictionary, key):
    """Extract a list from dictionary by key name."""
    if key in dictionary and isinstance(dictionary[key], list):
        return np.array(dictionary[key])
    return np.array([])


def get_ith_key_list(dictionary, key_idx):
    """Extract the list at the specified key index from a dictionary (legacy support)."""
    keys = list(dictionary.keys())
    if len(keys) >= key_idx:
        ith_key = keys[key_idx - 1]
        if isinstance(dictionary[ith_key], list):
            return np.array(dictionary[ith_key])
    return np.array([])


def get_similarity_scores(client, input_actions, reference_list):
    """Calculate similarity scores between input actions and reference actions using GPT-4."""
    if len(input_actions) == 0:
        return np.array([])

    reference_list_length = len(reference_list)
    input_actions_length = len(input_actions)

    prompt = f"""
    I have a list of behavioral actions {reference_list} as a reference.
    I want to predict the similarity of a list of input actions with the labels in the above reference list.
    Output should be an array of size ({input_actions_length} x {reference_list_length}) with a similarity score between 0 and 1.
    Similarity scores for a given input action should sum up to 1 and should not have same values.
    Each row of the array should indicate similarities for a single input action.
    Do not explain. Only output the array without any texts.

    The input actions are {input_actions.tolist() if isinstance(input_actions, np.ndarray) else input_actions}
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )

    similarity_scores_str = response.choices[0].message.content.strip()
    similarity_scores_list = eval(similarity_scores_str)
    similarity_scores_array = np.array(similarity_scores_list)

    return similarity_scores_array


def calculate_input_action_costs(similarity_scores, reference_costs):
    """Map behavioral actions to costs based on similarity to reference actions."""
    if similarity_scores.size == 0:
        return []

    # Handle 1D array (single action) vs 2D array (multiple actions)
    if similarity_scores.ndim == 1:
        most_similar_indices = [np.argmax(similarity_scores)]
    else:
        most_similar_indices = np.argmax(similarity_scores, axis=1)

    input_action_costs = [reference_costs[index] for index in most_similar_indices]
    return input_action_costs


def format_list(arr):
    """Format a numpy array or list for display."""
    if arr is None or (isinstance(arr, np.ndarray) and arr.size == 0):
        return "(none detected)"
    if isinstance(arr, np.ndarray):
        return ", ".join(f'"{item}"' for item in arr)
    return str(arr)


def print_results(instruction, landmark_list, navigation_action_list,
                  behavioral_action_list, behavioral_target_list,
                  input_action_costs, reference_actions, reference_costs):
    """Print formatted results to stdout."""

    separator = "=" * 70

    print(f"\n{separator}")
    print("INSTRUCTION DECOMPOSITION RESULTS")
    print(separator)

    print(f"\nInput Instruction:")
    print(f"  \"{instruction}\"")

    print(f"\n{'-' * 70}")
    print("EXTRACTED COMPONENTS")
    print(f"{'-' * 70}")

    print(f"\n  Landmarks:          {format_list(landmark_list)}")
    print(f"  Navigation Actions: {format_list(navigation_action_list)}")
    print(f"  Behavioral Actions: {format_list(behavioral_action_list)}")
    print(f"  Behavioral Targets: {format_list(behavioral_target_list)}")

    print(f"\n{'-' * 70}")
    print("BEHAVIORAL COST MAPPING")
    print(f"{'-' * 70}")

    print(f"\n  Reference Actions & Costs:")
    for action, cost in zip(reference_actions, reference_costs):
        print(f"    {action:10} -> {cost}")

    if behavioral_action_list is not None and len(behavioral_action_list) > 0 and len(input_action_costs) > 0:
        print(f"\n  Computed Costs for Input Actions:")
        print(f"  {'Action':<25} {'Assigned Cost':>15}")
        print(f"  {'-' * 40}")
        for action, cost in zip(behavioral_action_list, input_action_costs):
            print(f"  {action:<25} {cost:>15}")
    else:
        print("\n  No behavioral actions detected to compute costs.")

    print(f"\n{separator}\n")


def process_instruction(client, instruction):
    """Process a single instruction and print results."""
    print("\nProcessing instruction... (calling GPT-4)")

    try:
        # Get instruction breakdown
        instruction_breakdown = get_instruction_breakdown(client, instruction)

        # Extract lists using known keys (with fallback to index-based extraction)
        landmark_list = get_list_by_key(instruction_breakdown, 'landmarks')
        if landmark_list.size == 0:
            landmark_list = get_ith_key_list(instruction_breakdown, key_idx=1)

        navigation_action_list = get_list_by_key(instruction_breakdown, 'navigation_actions')
        if navigation_action_list.size == 0:
            navigation_action_list = get_ith_key_list(instruction_breakdown, key_idx=2)

        behavioral_action_list = get_list_by_key(instruction_breakdown, 'behavioral_actions')
        if behavioral_action_list.size == 0:
            behavioral_action_list = get_ith_key_list(instruction_breakdown, key_idx=3)

        behavioral_target_list = get_list_by_key(instruction_breakdown, 'behavioral_targets')
        if behavioral_target_list.size == 0:
            behavioral_target_list = get_ith_key_list(instruction_breakdown, key_idx=4)

        # Calculate behavioral costs
        input_action_costs = []
        if behavioral_action_list is not None and len(behavioral_action_list) > 0:
            similarity_scores = get_similarity_scores(client, behavioral_action_list, REFERENCE_ACTIONS)
            input_action_costs = calculate_input_action_costs(similarity_scores, REFERENCE_COSTS)

        # Print results
        print_results(
            instruction, landmark_list, navigation_action_list,
            behavioral_action_list, behavioral_target_list,
            input_action_costs, REFERENCE_ACTIONS, REFERENCE_COSTS
        )

        return True

    except Exception as e:
        print(f"\nError processing instruction: {e}")
        return False


def interactive_mode(client):
    """Run in interactive mode, prompting for instructions."""
    print("\n" + "=" * 70)
    print("BEHAV INSTRUCTION DECOMPOSITION - Interactive Mode")
    print("=" * 70)
    print("\nEnter navigation instructions to decompose them into components.")
    print("Type 'quit', 'exit', or 'q' to exit.\n")

    while True:
        try:
            instruction = input("Enter instruction: ").strip()

            if instruction.lower() in ['quit', 'exit', 'q', '']:
                if instruction == '':
                    print("Empty input. Type 'quit' to exit or enter an instruction.")
                    continue
                print("\nExiting. Goodbye!")
                break

            process_instruction(client, instruction)

        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            break
        except EOFError:
            print("\n\nEnd of input. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description="Decompose navigation instructions into landmarks, actions, targets, and behavioral costs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive              # Interactive mode
  %(prog)s -i                         # Interactive mode (short)
  %(prog)s --demo                     # Run with example instruction
  %(prog)s "Turn left at the tree"    # Process single instruction
  %(prog)s --instruction "Stay on sidewalk, avoid grass"

Environment Variables:
  OPENAI_API_KEY    Required. Your OpenAI API key for GPT-4.

Docker Usage:
  docker-compose --profile test run --rm instruction_decomposition \\
      python /app/instruction-decomposition/GPT4o/instruction-to-behavioral-costs.py -i
        """
    )

    parser.add_argument(
        'instruction',
        nargs='?',
        help='Navigation instruction to process (positional argument)'
    )
    parser.add_argument(
        '--instruction', '-I',
        dest='instruction_flag',
        help='Navigation instruction to process (named argument)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode, prompting for instructions'
    )
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run with a demo instruction to test the system'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show raw GPT-4 responses for debugging'
    )

    args = parser.parse_args()

    # Set verbose flag globally
    global VERBOSE
    VERBOSE = args.verbose

    # Initialize OpenAI client
    try:
        client = get_openai_client()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Determine mode
    if args.interactive:
        interactive_mode(client)
    elif args.demo:
        print("Running demo with example instruction...")
        process_instruction(client, DEMO_INSTRUCTION)
    elif args.instruction or args.instruction_flag:
        instruction = args.instruction or args.instruction_flag
        process_instruction(client, instruction)
    else:
        # Default: show help
        parser.print_help()
        print("\n" + "-" * 70)
        print("No instruction provided. Use --demo for a test run or -i for interactive mode.")
        sys.exit(0)


if __name__ == "__main__":
    main()
