#!/usr/bin/env python3
"""
Generate a report about the task allocation database.
"""

import json
from pathlib import Path
from typing import List, Dict, Tuple

def get_domain_directories(base_path: Path) -> List[Path]:
    """Get all domain directories."""
    domains = []
    for item in base_path.iterdir():
        if item.is_dir() and item.name not in ['.git', '__pycache__']:
            # Check if it has the expected structure
            if (item / 'tasks.txt').exists() and (item / 'agents.json').exists():
                domains.append(item)
    return sorted(domains)

def read_tasks(domain_path: Path) -> List[str]:
    """Read task names from tasks.txt."""
    tasks_file = domain_path / 'tasks.txt'
    if not tasks_file.exists():
        return []
    with open(tasks_file, 'r', encoding='utf-8') as f:
        tasks = [line.strip() for line in f if line.strip()]
    return tasks

def extract_task_type_keywords(task_name: str) -> str:
    """Extract 1-2 word keywords from task name."""
    words = task_name.split()
    if not words:
        return task_name
    
    # Common verbs to skip
    skip_verbs = ['run', 'build', 'prepare', 'draft', 'revise', 'package', 'design', 
                  'implement', 'analyze', 'summarize', 'define', 'cook', 'prep', 
                  'assemble', 'receive', 'store', 'update', 'check', 'collect', 
                  'gather', 'start', 'set', 'scout', 'plan', 'repair', 'organize', 
                  'pack', 'navigate']
    
    # If first word is a verb to skip, take next 1-2 words
    if words[0].lower() in skip_verbs:
        if len(words) >= 3:
            return ' '.join(words[1:3])
        elif len(words) >= 2:
            return words[1]
        else:
            return words[0]
    
    # Otherwise take first 1-2 words
    if len(words) >= 2:
        return ' '.join(words[:2])
    return words[0]

def read_resources(domain_path: Path) -> Tuple[List[str], List[str]]:
    """Read private and public resources."""
    private_file = domain_path / 'private_resources.txt'
    public_file = domain_path / 'public_resources.txt'
    
    private = []
    if private_file.exists():
        with open(private_file, 'r', encoding='utf-8') as f:
            private = [line.strip() for line in f if line.strip()]
    
    public = []
    if public_file.exists():
        with open(public_file, 'r', encoding='utf-8') as f:
            public = [line.strip() for line in f if line.strip()]
    
    return private, public

def read_agents(domain_path: Path) -> Dict:
    """Read agents.json and return statistics."""
    agents_file = domain_path / 'agents.json'
    if not agents_file.exists():
        return {
            'total': 0,
            'leaders': 0,
            'workers': 0,
            'both': 0
        }
    
    with open(agents_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    agent_pool = data.get('agent_pool', [])
    total = len(agent_pool)
    leaders = 0
    workers = 0
    both = 0
    
    for agent in agent_pool:
        roles = agent.get('possible_roles', [])
        can_lead = 'leader' in roles
        can_work = 'worker' in roles
        
        if can_lead and can_work:
            both += 1
            leaders += 1
            workers += 1
        elif can_lead:
            leaders += 1
        elif can_work:
            workers += 1
    
    return {
        'total': total,
        'leaders': leaders,
        'workers': workers,
        'both': both
    }

def generate_report(base_path: Path) -> str:
    """Generate the complete report."""
    domains = get_domain_directories(base_path)
    
    report_lines = []
    report_lines.append("# Task Allocation Database Report\n")
    
    # Section A: Task allocation assets
    report_lines.append("## A. Task allocation assets\n")
    report_lines.append(f"How many task allocation domains?\n")
    report_lines.append(f"{len(domains)} (research project collaboration, cafeteria operations, camping in the wild)\n")
    report_lines.append("")
    
    for domain_path in domains:
        domain_name = domain_path.name.replace('_', ' ').title()
        report_lines.append(f"### {domain_name}\n")
        
        # Tasks
        tasks = read_tasks(domain_path)
        task_count = len(tasks)
        report_lines.append(f"**Number of task templates:** {task_count}\n")
        
        # Example task types (1-2 words)
        if tasks:
            example_types = [extract_task_type_keywords(task) for task in tasks[:5]]
            example_str = ', '.join(example_types)
            report_lines.append(f"**Example task types:** {example_str}\n")
        
        # Resources
        private_resources, public_resources = read_resources(domain_path)
        report_lines.append(f"**Resource types:**\n")
        report_lines.append(f"- List of private resources: {', '.join(private_resources)}\n")
        report_lines.append(f"- List of public resources: {', '.join(public_resources)}\n")
        
        # Agent pool
        agent_stats = read_agents(domain_path)
        report_lines.append(f"**Agent pool:**\n")
        report_lines.append(f"- Total number of agent personas: {agent_stats['total']}\n")
        report_lines.append(f"- How many can be leaders: {agent_stats['leaders']}\n")
        report_lines.append(f"- How many workers per instance: 2 (fixed)\n")
        report_lines.append("")
    
    # Section B: Instance generation process
    report_lines.append("## B. Instance generation process\n")
    report_lines.append("For each task allocation instance:\n")
    report_lines.append("")
    report_lines.append("**How many tasks are sampled per instance?**\n")
    report_lines.append("~10 tasks (sampled from the domain's task pool)\n")
    report_lines.append("")
    report_lines.append("**How many agents per instance?**\n")
    report_lines.append("3 agents (1 Leader + 2 Workers)\n")
    report_lines.append("")
    report_lines.append("**How are agent–task efficiencies generated?**\n")
    report_lines.append("Efficiency values are pre-defined in the database for each (persona, task) pair. Each persona has an efficiency value (typically in range [0.4, 1.6]) for every task in the domain. These values are fixed in `agents.json` and reflect the persona's strengths and weaknesses.\n")
    report_lines.append("")
    report_lines.append("**How are private/public resource budgets generated?**\n")
    report_lines.append("- **Private resource capacities**: For each agent and private resource, sampled from the persona's capacity distribution (Gaussian with `mean` and `std`, truncated to non-negative). Each persona defines distribution parameters for all private resources in `agents.json`.\n")
    report_lines.append("- **Public resource budgets**: Fixed team-level budgets (not sampled per agent, as public resources are shared). The game-generation program sets these budgets based on the selected tasks' public resource requirements.\n")
    report_lines.append("")
    report_lines.append("**How is partial observability enforced?**\n")
    report_lines.append("Each agent has private information that others cannot observe:\n")
    report_lines.append("- **Private resource capacities**: Each agent only knows their own private resource capacities (sampled from their persona's distribution). Other agents' private capacities are not visible.\n")
    report_lines.append("- **Task values**: All agents can see task values (shared information).\n")
    report_lines.append("- **Efficiency values**: All agents can see the efficiency matrix (which persona is good at which tasks).\n")
    report_lines.append("- **Public resource costs**: All agents can see baseline public resource requirements for tasks.\n")
    report_lines.append("")
    report_lines.append("**What does each agent not see?**\n")
    report_lines.append("- Other agents' private resource capacities (each agent only knows their own)\n")
    report_lines.append("- Other agents' exact persona identity may be partially hidden depending on implementation\n")
    report_lines.append("")
    
    # Section C: Dataset scale
    report_lines.append("## C. Dataset scale\n")
    report_lines.append("**Number of task allocation instances used in experiments:**\n")
    report_lines.append("(To be determined by the game-generation program; not specified in the database)\n")
    report_lines.append("")
    report_lines.append("**Whether different instances share:**\n")
    report_lines.append("- **Task templates**: Yes, all instances from the same domain share the same task template pool (defined in `tasks.txt`). Each instance samples ~10 tasks from this pool.\n")
    report_lines.append("- **Agents**: Yes, all instances from the same domain share the same agent persona pool (defined in `agents.json`). Each instance samples 3 personas from this pool (1 leader + 2 workers).\n")
    report_lines.append("- **Resources**: Yes, all instances from the same domain share the same resource types (defined in `resources.txt`, `private_resources.txt`, `public_resources.txt`).\n")
    report_lines.append("")
    
    # Section D: Evaluation alignment
    report_lines.append("## D. Evaluation alignment (quick check)\n")
    report_lines.append("**Is there:**\n")
    report_lines.append("- only one task allocation setting?\n")
    report_lines.append("- or multiple variants (e.g., with/without multipliers)?\n")
    report_lines.append("")
    report_lines.append("The database supports an optional variant: public resource cost multipliers can be enabled or disabled via the `use_public_resource_multipliers` parameter in the game-generation program. When enabled (`use_public_resource_multipliers=True`), each agent's public resource costs are multiplied by their persona's `public_resource_cost_multipliers` (default 1.0 if not specified). When disabled (default `False`), all agents use the same baseline public resource costs. This creates two evaluation variants:\n")
    report_lines.append("1. **Baseline variant** (default): All agents use identical baseline public resource costs\n")
    report_lines.append("2. **Multiplier variant**: Agents have different public resource costs based on their persona multipliers\n")
    report_lines.append("")
    
    return '\n'.join(report_lines)

def main():
    """Main entry point."""
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    base_path = script_dir
    
    # Generate report
    report = generate_report(base_path)
    
    # Print to stdout
    print(report)
    
    # Also save to file
    output_file = base_path / 'database_report.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nReport also saved to: {output_file}")

if __name__ == '__main__':
    main()

