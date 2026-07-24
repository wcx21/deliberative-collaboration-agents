#!/usr/bin/env python3
"""
Token usage statistics for each game under a given result path.

Usage:
    python stat_token_usage.py <result_path>

Example:
    python scripts/evaluation/token_usage.py result/debug_experiment
"""

import os
import re
import sys
from pathlib import Path
from collections import defaultdict
from delib_collab.paths import PROJECT_ROOT


def find_token_log_files(result_path):
    """
    Find token log files in the logs directory corresponding to the result path.
    
    Args:
        result_path: Path to result directory, e.g. /path/to/result/exp_name
        
    Returns:
        dict: {game_id: token_log_file_path}
    """
    result_path = Path(result_path).resolve()
    
    if 'result' in str(result_path):
        logs_path = str(result_path).replace('result', 'logs')
    else:
        logs_path = result_path.parent.parent / 'logs' / result_path.name
    
    logs_path = Path(logs_path)
    
    if not logs_path.exists():
        possible_logs_paths = [
            result_path.parent.parent / 'logs' / result_path.name,
            result_path.parent / 'logs' / result_path.name,
            PROJECT_ROOT / 'logs' / result_path.name,
        ]
        for p in possible_logs_paths:
            if p.exists():
                logs_path = p
                break
        else:
            raise FileNotFoundError(f"Cannot find corresponding logs directory: {result_path}")
    
    print(f"Searching for token log files in: {logs_path}")
    
    token_files = {}
    for token_file in logs_path.rglob("*token*.log"):
        match = re.search(r'game(\d+)_token', token_file.name)
        if match:
            game_id = int(match.group(1))
            token_files[game_id] = token_file
    
    return token_files, logs_path


def parse_token_log(token_file):
    """
    Parse a token log file and extract token usage statistics for all API calls.
    
    Args:
        token_file: Path to the token log file
        
    Returns:
        tuple: (total_input_tokens, total_output_tokens, num_calls)
    """
    total_input = 0
    total_output = 0
    num_calls = 0
    
    try:
        with open(token_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        input_pattern = r'Total API input tokens:\s*(\d+)'
        output_pattern = r'Total API output tokens:\s*(\d+)'
        
        input_matches = re.findall(input_pattern, content)
        output_matches = re.findall(output_pattern, content)
        
        for input_val in input_matches:
            total_input += int(input_val)
            num_calls += 1
        
        for output_val in output_matches:
            total_output += int(output_val)
        
        if len(input_matches) != len(output_matches):
            print(f"  Warning: {token_file.name} has mismatched input/output record counts "
                  f"(input: {len(input_matches)}, output: {len(output_matches)})")
            
    except Exception as e:
        print(f"  Error: cannot parse {token_file}: {e}")
        return 0, 0, 0
    
    return total_input, total_output, num_calls


def stat_token_usage(result_path):
    """
    Compute token usage statistics for all games under the given result path.
    
    Args:
        result_path: Path to result directory
    """
    result_path = Path(result_path)
    
    if not result_path.exists():
        print(f"Error: path does not exist: {result_path}")
        return
    
    print("=" * 80)
    print(f"Token Usage Statistics: {result_path}")
    print("=" * 80)
    
    try:
        token_files, logs_path = find_token_log_files(result_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    if not token_files:
        print(f"No token log files found")
        return
    
    print(f"\nFound {len(token_files)} game token log files\n")
    
    game_stats = {}
    for game_id in sorted(token_files.keys()):
        token_file = token_files[game_id]
        print(f"Parsing game_{game_id}: {token_file.name}")
        
        input_tokens, output_tokens, num_calls = parse_token_log(token_file)
        total_tokens = input_tokens + output_tokens
        
        game_stats[game_id] = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': total_tokens,
            'num_calls': num_calls
        }
        
        print(f"  Input tokens:  {input_tokens:,}")
        print(f"  Output tokens: {output_tokens:,}")
        print(f"  Total tokens:  {total_tokens:,}")
        print(f"  API calls:     {num_calls}")
        print()
    
    if not game_stats:
        print("No valid token statistics data")
        return
    
    total_input_all = sum(s['input_tokens'] for s in game_stats.values())
    total_output_all = sum(s['output_tokens'] for s in game_stats.values())
    total_tokens_all = sum(s['total_tokens'] for s in game_stats.values())
    total_calls_all = sum(s['num_calls'] for s in game_stats.values())
    num_games = len(game_stats)
    
    avg_input = total_input_all / num_games
    avg_output = total_output_all / num_games
    avg_total = total_tokens_all / num_games
    avg_calls = total_calls_all / num_games
    
    print("=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(f"Number of games: {num_games}")
    print(f"\nTotal:")
    print(f"  Input tokens:  {total_input_all:,}")
    print(f"  Output tokens: {total_output_all:,}")
    print(f"  Total tokens:  {total_tokens_all:,}")
    print(f"  API calls:     {total_calls_all}")
    print(f"\nPer Game Average:")
    print(f"  Input tokens:  {avg_input:,.0f}")
    print(f"  Output tokens: {avg_output:,.0f}")
    print(f"  Total tokens:  {avg_total:,.0f}")
    print(f"  API calls:     {avg_calls:.1f}")
    print("=" * 80)
    
    print("\nPer-Game Detailed Statistics:")
    print("-" * 80)
    print(f"{'Game ID':<10} {'Input Tokens':<15} {'Output Tokens':<15} {'Total Tokens':<15} {'API Calls':<10}")
    print("-" * 80)
    for game_id in sorted(game_stats.keys()):
        stats = game_stats[game_id]
        print(f"game_{game_id:<6} {stats['input_tokens']:>13,} {stats['output_tokens']:>13,} "
              f"{stats['total_tokens']:>13,} {stats['num_calls']:>8}")
    print("=" * 80)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"\nError: please provide a result path")
        print(f"Example: python {sys.argv[0]} /path/to/result/exp_name")
        sys.exit(1)
    
    result_path = sys.argv[1]
    stat_token_usage(result_path)
