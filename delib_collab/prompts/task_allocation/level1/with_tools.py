#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Task allocation Level 1 prompt templates (with tools version)."""

TASK_DESCRIPTION = """
You are an AI agent in a task allocation scenario. Your role is {agent_name}.

The team has 3 members:
- agent_0 (Worker): A team member who can execute tasks
- agent_1 (Worker): A team member who can execute tasks  
- agent_2 (Leader): The team leader who has the final decision authority,can also execute tasks

Your goal is to collaboratively decide how to allocate {num_tasks} tasks among the 3 team members.

The task names you MUST use as JSON keys (do NOT use task_1/task_2 aliases): {task_names}

Key information:
1. **Task Allocation**: Each task should be assigned to exactly one agent (or left unassigned if impossible due to resource constraints). The allocation is a mapping like {{"task_1": "agent_0", "task_2": "agent_1", ...}}. Note: Tasks can be left unassigned if there are insufficient resources to complete them.
2. **Resources**: 
   - Private resources: Each agent has their own private resources (e.g., Time, GPU) that only they can use.
   - Public resources: The team shares some common resources (e.g., Budget) that any task can consume.
   - A task can only be assigned to an agent if they have sufficient resources.

3. **Efficiency Values**: Each agent has different efficiency for different tasks. The reward for assigning a task to an agent equals their efficiency value for that task.
4. **Total Reward**: The team's total reward is the sum of efficiency values for all task assignments. You want to maximize this.

5. **Partial Observation**: You can only observe part of the resources and efficiency values. Other agents may observe different parts. You need to communicate to learn the full picture.

You can communicate for {max_round} rounds. In each round, you can say up to {max_character} characters.

**Decision Authority**:
- agent_0 and agent_1 (Workers): Can propose allocations and indicate acceptance, but cannot finalize the agreement.
- agent_2 (Leader): Has the final decision authority. Only the Leader's final_decision can end the negotiation.

To collaborate, we suggest you do the following steps in each round:
1. Estimate the current state of resources and efficiency values
2. Think about the best task allocation
3. Decide whether to accept your partners' proposals
4. Talk to your partners

"""



OBSERVE_PROMPT = '''You are {agent_name} in a 3-agent team allocating {num_tasks} tasks. Goal: maximize total value under resource constraints.

== SCENARIO OVERVIEW ==

In a team of 3 agents, you (along with 2 partners) need to collaboratively allocate {num_tasks} tasks. 
Each agent has:
- **Private resources**: Resources that only this agent knows about and can use
- **Public resources**: Shared resources that all agents can observe

Each task has specific resource requirements and generates different value when completed by different agents.
The goal is to reach a task allocation that maximizes total team value while satisfying all resource constraints.

== YOUR ROLE ==

You are {agent_name}. **Note**: If you are agent_2, you are the Leader with final decision authority. If you are agent_0 or agent_1, you are a Worker.

In this phase, your job is to:
1. Infer what resources and preferences your partners might have based on the conversation
2. Combine this with your own resources to estimate the total available resources
3. Combine preferences to get an overall picture of task values

== YOUR PRIVATE INFORMATION ==

<your_private_resources>
{current_private_resources}
</your_private_resources>

<public_resources_you_observe>
{current_public_resources}
</public_resources_you_observe>

<your_task_efficiency>
{efficiency_dict}
</your_task_efficiency>

== TASK REQUIREMENTS(resources needed to complete each task, same for all agents) ==
(Note: Missing resource = 0 required)
<task_requirements>
{task_requirements}
</task_requirements>


== CONVERSATION HISTORY ==
<conversation_history>
{chat_history}
</conversation_history>

== YOUR TASK ==
Infer from conversation history and output:

1. **partner_resources**: Partners' private resources (format: {{"agent_X": {{"resource": amount}}}})
2. **partner_preferences**: Partners' efficiency values (format: {{"agent_X": {{"task": efficiency}}}})
3. **total_resources**: Combined resources
   - **CRITICAL**: Private resources stay SEPARATE per agent (do NOT add). Public resources MUST be ADDED: total_public = your_obs + partner1_obs + partner2_obs
   - **IMPORTANT**: When checking resource constraints later, you must use this TOTAL pooled public resources (sum of all agents' observations), NOT just your own observed portion. Public resources are shared among all agents, so the constraint check uses the combined total.
4. **overall_preferences**: All agents' efficiency values (format: {{"agent_X": {{"task": efficiency}}}})

**Output Format**: Use XML tags, JSON inside (NO markdown code blocks):
<thinking_process_1>
reasoning...
</thinking_process_1>
<partner_resources>
{{"agent_X": {{"resource": amount}}}}
</partner_resources>
<partner_preferences>
{{"agent_X": {{"task": efficiency}}}}
</partner_preferences>
<thinking_process_2>
reasoning...
</thinking_process_2>
<total_resources>
{{"agent_private_resources": {{"agent_0": {{...}}, "agent_1": {{...}}, "agent_2": {{...}}}}, "public_resources": {{...}}}}
</total_resources>
<overall_preferences>
{{"agent_0": {{"task": efficiency}}, "agent_1": {{...}}, "agent_2": {{...}}}}
</overall_preferences>

**Key Rules**: Round 1 → make reasonable guesses. Later → base on conversation. Always output numeric values (no "unknown"). Be specific about resource types and amounts. Private=separate, Public=sum.
'''



THINK_PROMPT = '''You are {agent_name}, participating in a task allocation negotiation.

== CURRENT ROUND ==
Round {round_count} of {max_round}

== SCENARIO RECAP ==

You and 2 partners need to allocate {num_tasks} tasks among 3 agents.
The goal is to maximize total team value while satisfying resource constraints.

**Your Role**: If you are agent_2, you are the Leader with final decision authority. If you are agent_0 or agent_1, you are a Worker.

== YOUR OWN OBSERVATIONS ==
**IMPORTANT**: These are what you directly observed, not estimates:

<your_private_resources>
{current_private_resources}
</your_private_resources>

<your_public_resources_observation>
{current_public_resources}
</your_public_resources_observation>

<your_task_efficiency>
{efficiency_dict}
</your_task_efficiency>

== YOUR ESTIMATIONS FROM PREVIOUS PHASE ==
Based on your analysis, you have estimated:

<total_resources_estimation>
{total_resources}
</total_resources_estimation>

<overall_preferences_estimation>
{overall_preferences}
</overall_preferences_estimation>

== TASK REQUIREMENTS(resources needed to complete each task, same for all agents) ==
**Note**: If a resource is not listed for a task, it means that task requires 0 of that resource.

<task_requirements>
{task_requirements}
</task_requirements>

Here is the feasibility analysis - which tasks can be assigned to which agents:
<available_allocations>
{available_allocations}
</available_allocations>

Here are the tasks that cannot be assigned to certain agents due to insufficient resources:
<insufficient_allocation_info>
{insufficient_allocation_info}
</insufficient_allocation_info>

Here is the optimal allocation based on total resources and overall preferences:
<best_allocation>
{best_allocation}
</best_allocation>

Note: the information above is based on your estimation so far. It may not be the ground truth. The tool's results are reliable only if your estimated resources and efficiency values are accurate.

== YOUR PRIVATE INFORMATION (for reference) ==

<your_private_resources>
{current_private_resources}
</your_private_resources>

<your_task_efficiency>
{efficiency_dict}
</your_task_efficiency>

== YOUR TASK (THINK PHASE) ==

Based on your observations and estimations, propose a task allocation that:
1. Satisfies resource constraints (each agent has enough resources for their assigned tasks)
2. Maximizes total team value (assign tasks to agents who are most efficient at them)
3. Is feasible given the total resources available

**CRITICAL - Resource Constraint Verification:**
Before proposing, verify: (1) Each agent's private resource demands ≤ their capacity, (2) Total public resource demand ≤ TOTAL pooled public resources (sum of all agents' observations). If constraints violated, adjust allocation.

**Output Format**: Use XML tags. JSON inside tags (no markdown code blocks).

<reasoning_process>
(Your overall reasoning about the allocation)
</reasoning_process>

<proposal>
{{
    "allocation_proposal": {{
        "task_name_1": "agent_X",
        "task_name_2": "agent_Y",
        ...
    }},
    "explanation": "Your reasoning for this allocation, including confirmation that resource constraints are satisfied..."
}}
</proposal>

IMPORTANT:
- Each task should be assigned to exactly one agent (or left unassigned if impossible)
- Verify that your proposal does NOT exceed available resources
- Explain your reasoning clearly to help reach consensus
'''



DECISION_PROMPT = '''You are {agent_name}, participating in a task allocation negotiation.

== CURRENT ROUND ==
Round {round_count} of {max_round}

== PARTNER'S PROPOSAL ==

Your partner has proposed the following allocation:

<partner_proposal>
{partner_proposal}
</partner_proposal>

<partner_explanation>
{partner_explanation}
</partner_explanation>

== YOUR OWN PROPOSAL ==

You have proposed:

<your_proposal>
{agent_proposal}
</your_proposal>

<your_explanation>
{agent_proposal_explanation}
</your_explanation>

== YOUR OWN OBSERVATIONS ==
**IMPORTANT**: These are what you directly observed, not estimates:

<your_private_resources>
{current_private_resources}
</your_private_resources>

<your_public_resources_observation>
{current_public_resources}
</your_public_resources_observation>

<your_task_efficiency>
{efficiency_dict}
</your_task_efficiency>

== YOUR ESTIMATIONS ==

<total_resources_estimation>
{total_resources}
</total_resources_estimation>

<overall_preferences_estimation>
{overall_preferences}
</overall_preferences_estimation>

Based on your estimation:
Here is the feasibility analysis - which tasks can be assigned to which agents:
<available_allocations>
{available_allocations}
</available_allocations>

Here are the tasks that cannot be assigned to certain agents due to insufficient resources:
<insufficient_allocation_info>
{insufficient_allocation_info}
</insufficient_allocation_info>

Here is the optimal allocation based on total resources and overall preferences:
<best_allocation>
{best_allocation}
</best_allocation>

Note: the information above is based on your estimation so far. It may not be the ground truth. The tool's results are reliable only if your estimated resources and efficiency values are accurate.

== TASK REQUIREMENTS(resources needed to complete each task, same for all agents) ==
**Note**: If a resource is not listed for a task, it means that task requires 0 of that resource.


<task_requirements>
{task_requirements}
</task_requirements>

== YOUR TASK (DECISION PHASE) ==

Evaluate whether to accept your partner's proposal.



Consider:
1. **Does the proposal satisfy resource constraints?** 
2. Does it provide good value for all team members?
3. Is it fair to you given your efficiency values?
4. How does it compare to your own proposal?

**If you are a Worker (agent_0 or agent_1):**
- Set `accept_decision` to indicate if you would accept this proposal
- Your decision is advisory; only the Leader can make the final call
- Workers should NOT set `final_decision` field

**If you are the Leader (agent_2):**
- You have the authority to make the FINAL decision
- You can either:
  * Continue negotiating: Don't include a final allocation in your response
  * End negotiation NOW: Include the final allocation you want to implement
- **IMPORTANT**: If you think the current proposal (yours or your partners') is good enough, you should make the final decision to end the negotiation. Don't wait indefinitely!

**CRITICAL - Output Format Requirements**:
1. You MUST use XML tags with angle brackets (e.g., <reasoning_process>...</reasoning_process>)
2. Inside the tags that require JSON, output JSON DIRECTLY without markdown code blocks
3. NO ```json or ``` markers around the JSON

Please respond in the following format:

<reasoning_process>
(Your reasoning about the decision)
</reasoning_process>

<decision>
{{
    "accept_decision": true or false,
    "explanation": "Your reasoning (including resource constraint consideration)...",
    "final_decision": (Only for Leader) The allocation dict if you want to finalize, or leave this field out if you want to continue
}}
</decision>

**Examples:**

Worker's decision (no final_decision field):
{{"accept_decision": true, "explanation": "I agree with this proposal"}}

Leader continues negotiating (no final_decision field):
{{"accept_decision": false, "explanation": "I need more information"}}

Leader makes final decision:
{{"accept_decision": true, "explanation": "This is optimal", "final_decision": {{"task_A": "agent_0", "task_B": "agent_1"}}}}
'''



TALK_PROMPT = '''You are {agent_name}, participating in a task allocation negotiation.

== CURRENT ROUND ==
Round {round_count} of {max_round}

**Your Role**: If you are agent_2, you are the Leader with final decision authority. If you are agent_0 or agent_1, you are a Worker.

== CONVERSATION HISTORY ==

<conversation_history>
{chat_history}
</conversation_history>

== YOUR PROPOSAL ==

<your_proposal>
{agent_proposal}
</your_proposal>

<your_explanation>
{agent_proposal_explanation}
</your_explanation>

== YOUR DECISION ON PARTNER'S PROPOSAL ==

<your_decision>
Accept: {accept_decision}
Reason: {decision_explanation}
</your_decision>

== YOUR ESTIMATIONS ==

<total_resources_estimation>
{total_resources}
</total_resources_estimation>

<overall_preferences_estimation>
{overall_preferences}
</overall_preferences_estimation>

Here is the feasibility analysis - which tasks can be assigned to which agents:
<available_allocations>
{available_allocations}
</available_allocations>

Here are the tasks that cannot be assigned to certain agents due to insufficient resources:
<insufficient_allocation_info>
{insufficient_allocation_info}
</insufficient_allocation_info>

Here is the optimal allocation based on total resources and overall preferences:
<best_allocation>
{best_allocation}
</best_allocation>

Note: the information above is based on your estimation so far. It may not be the ground truth. The tool's results are reliable only if your estimated resources and efficiency values are accurate.

== YOUR OWN OBSERVATIONS ==
**IMPORTANT**: These are what you directly observed, not estimates:

<your_private_resources>
{current_private_resources}
</your_private_resources>

<your_public_resources_observation>
{current_public_resources}
</your_public_resources_observation>

<your_task_efficiency>
{efficiency_dict}
</your_task_efficiency>

== YOUR TASK (TALK PHASE) ==

Now, for the next round, please organize what to talk with your partners.
*Important*: You can say up to {max_character} characters, please carefully design your message.

Here are some suggestions (you are not required to follow all of them):
- It is very important to share your observation, including your private resources, public resources observation, and task efficiency values.
- Especially you should share the resources you have and your efficiency values for different tasks.
- You may want to explain why you don't agree with your partner's proposal or state your agreement if the proposal is satisfactory.

**Output Format**: Use XML tags. Output JSON directly (no markdown code blocks).

<reasoning_process>
(Your reasoning)
</reasoning_process>

<message_content>
(Your message - be concise, focus on key updates and your proposal)
</message_content>
'''

# ========================= FULL-OBS VARIANTS (with_tools) =========================

# ========================= TASK_DESCRIPTION_FULL_OBS (with_tools) =========================
TASK_DESCRIPTION_FULL_OBS = "Full-Obs Deliberation Ablation Study With Tools (aligned with Baseline Oracle prompt format)"

# ========================= THINK_PROMPT_FULL_OBS (with_tools) =========================
THINK_PROMPT_FULL_OBS = '''You are {agent_name}, participating in a task allocation scenario.

== FULL OBSERVATION EXPERIMENT ==
This is a full observation experiment where you have access to ALL information from all agents.

== SCENARIO ==
You and 2 partners (agent_0 and agent_1) need to allocate {num_tasks} tasks among 3 agents.
The goal is to maximize total team value while satisfying resource constraints.

**Your Role**: If you are agent_2, you are the Leader with final decision authority. If you are agent_0 or agent_1, you are a Worker.

== CURRENT ROUND ==
Round {round_count} of {max_round}

The task names you MUST use as JSON keys: {task_names}

== COMPLETE INFORMATION FROM ALL AGENTS ==

**Agent 0 (Worker1) Efficiency values:**

<agent_0_efficiency_values>
{agent_0_efficiency}
</agent_0_efficiency_values>

**Agent 1 (Worker2) Efficiency values:**

<agent_1_efficiency_values>
{agent_1_efficiency}
</agent_1_efficiency_values>

**Agent 2 (Leader) Efficiency values:**

<agent_2_efficiency_values>
{agent_2_efficiency}
</agent_2_efficiency_values>

== TOTAL RESOURCES ==
**CRITICAL**: 
- Private resources stay SEPARATE per agent (do NOT add them together)
- Public resources MUST be ADDED: total_public = agent_0_obs + agent_1_obs + agent_2_obs

<total_private_resources>
Agent 0: {agent_0_private_resources}
Agent 1: {agent_1_private_resources}
Agent 2: {agent_2_private_resources}
</total_private_resources>

<total_public_resources>
{total_public_resources}
</total_public_resources>

== TASK REQUIREMENTS ==
<task_requirements>
{task_requirements}
</task_requirements>
(Note: Missing resource = 0 required)

Here is the feasibility analysis - which tasks can be assigned to which agents:
<available_allocations>
{available_allocations}
</available_allocations>

Here are the tasks that cannot be assigned to certain agents due to insufficient resources:
<insufficient_allocation_info>
{insufficient_allocation_info}
</insufficient_allocation_info>

Here is the optimal allocation based on total resources and overall preferences:
<best_allocation>
{best_allocation}
</best_allocation>

Note: the information above is based on your estimation so far. It may not be the ground truth. The tool's results are reliable only if your estimated resources and efficiency values are accurate.

== YOUR TASK (THINK PHASE) ==
Based on the complete information above, propose a task allocation.

**CRITICAL - Resource Constraint Verification:**
Before proposing, verify: (1) Each agent's private resource demands ≤ their capacity, (2) Total public resource demand ≤ TOTAL pooled public resources (sum of all agents' observations). If constraints violated, adjust allocation.

**IMPORTANT**:
- Each task should be assigned to exactly one agent (or left unassigned if impossible due to resource constraints)
- Verify that your proposal does NOT exceed available resources
- Maximize total team value (assign tasks to agents who are most efficient at them)

**Output Format**: Use XML tags. JSON inside tags (no markdown code blocks).

<reasoning_process>
(Your overall reasoning about the allocation, including resource constraint verification)
</reasoning_process>

<proposal>
{{
    "allocation_proposal": {{
        "task_name_1": "agent_X",
        "task_name_2": "agent_Y",
        ...
    }},
    "explanation": "Your reasoning for this allocation, including confirmation that resource constraints are satisfied..."
}}
</proposal>

IMPORTANT:
- Each task should be assigned to exactly one agent (or left unassigned if impossible)
- Verify that your proposal does NOT exceed available resources
- Explain your reasoning clearly to help reach consensus
'''


# ========================= DECISION_PROMPT_FULL_OBS (with_tools) =========================
DECISION_PROMPT_FULL_OBS = '''You are {agent_name}, participating in a task allocation scenario.

== FULL OBSERVATION EXPERIMENT ==
This is a full observation experiment where you have access to ALL information from all agents.

== CURRENT ROUND ==
Round {round_count} of {max_round}

The task names you MUST use as JSON keys: {task_names}

== PARTNER'S PROPOSAL ==

Your partner has proposed the following allocation:

<partner_proposal>
{partner_proposal}
</partner_proposal>

<partner_explanation>
{partner_explanation}
</partner_explanation>

== YOUR OWN PROPOSAL ==

You have proposed:

<your_proposal>
{agent_proposal}
</your_proposal>

<your_explanation>
{agent_proposal_explanation}
</your_explanation>

== COMPLETE INFORMATION FROM ALL AGENTS ==

**Agent 0 (Worker1) Efficiency values:**

<agent_0_efficiency_values>
{agent_0_efficiency}
</agent_0_efficiency_values>

**Agent 1 (Worker2) Efficiency values:**

<agent_1_efficiency_values>
{agent_1_efficiency}
</agent_1_efficiency_values>

**Agent 2 (Leader) Efficiency values:**

<agent_2_efficiency_values>
{agent_2_efficiency}
</agent_2_efficiency_values>

== TOTAL RESOURCES ==
**CRITICAL**: 
- Private resources stay SEPARATE per agent (do NOT add them together)
- Public resources MUST be ADDED: total_public = agent_0_obs + agent_1_obs + agent_2_obs

<total_private_resources>
Agent 0: {agent_0_private_resources}
Agent 1: {agent_1_private_resources}
Agent 2: {agent_2_private_resources}
</total_private_resources>

<total_public_resources>
{total_public_resources}
</total_public_resources>

== TASK REQUIREMENTS ==
<task_requirements>
{task_requirements}
</task_requirements>
(Note: Missing resource = 0 required)

Here is the feasibility analysis - which tasks can be assigned to which agents:
<available_allocations>
{available_allocations}
</available_allocations>

Here are the tasks that cannot be assigned to certain agents due to insufficient resources:
<insufficient_allocation_info>
{insufficient_allocation_info}
</insufficient_allocation_info>

Here is the optimal allocation based on total resources and overall preferences:
<best_allocation>
{best_allocation}
</best_allocation>

Note: the information above is based on your estimation so far. It may not be the ground truth. The tool's results are reliable only if your estimated resources and efficiency values are accurate.

== YOUR TASK (DECISION PHASE) ==

Evaluate whether to accept your partner's proposal.

**CRITICAL - Resource Constraint Verification:**
Before proposing, verify: (1) Each agent's private resource demands ≤ their capacity, (2) Total public resource demand ≤ TOTAL pooled public resources (sum of all agents' observations). If constraints violated, adjust allocation.

Consider:
1. **Does the proposal satisfy resource constraints?** (YOU MUST CALCULATE THIS EXPLICITLY)
2. Does it provide good value for all team members?
3. Is it fair to you given your efficiency values?
4. How does it compare to your own proposal?

**If you are a Worker (agent_0 or agent_1):**
- Set `accept_decision` to indicate if you would accept this proposal
- Your decision is advisory; only the Leader can make the final call
- Workers should NOT set `final_decision` field

**If you are the Leader (agent_2):**
- You have the authority to make the FINAL decision
- You can either:
  * Continue negotiating: Don't include a final allocation in your response
  * End negotiation NOW: Include the final allocation you want to implement
- **IMPORTANT**: If you think the current proposal (yours or your partners') is good enough, you should make the final decision to end the negotiation. Don't wait indefinitely!

**CRITICAL - Output Format Requirements**:
1. You MUST use XML tags with angle brackets (e.g., <reasoning_process>...</reasoning_process>)
2. Inside the tags that require JSON, output JSON DIRECTLY without markdown code blocks
3. NO ```json or ``` markers around the JSON

Please respond in the following format:

<reasoning_process>
(Your reasoning about the decision, including resource constraint verification)
</reasoning_process>

<decision>
{{
    "accept_decision": true or false,
    "explanation": "Your reasoning (MUST mention resource constraint verification result)...",
    "final_decision": (Only for Leader) The allocation dict if you want to finalize, or leave this field out if you want to continue
}}
</decision>

**Examples:**

Worker's decision (no final_decision field):
{{"accept_decision": true, "explanation": "I agree with this proposal"}}

Leader continues negotiating (no final_decision field):
{{"accept_decision": false, "explanation": "I need more information"}}

Leader makes final decision:
{{"accept_decision": true, "explanation": "This is optimal", "final_decision": {{"task_A": "agent_0", "task_B": "agent_1"}}}}
'''


# ========================= TALK_PROMPT_FULL_OBS (with_tools) =========================
TALK_PROMPT_FULL_OBS = '''You are {agent_name}, participating in a task allocation scenario.

== FULL OBSERVATION EXPERIMENT ==
This is a full observation experiment where you have access to ALL information from all agents.

== CURRENT ROUND ==
Round {round_count} of {max_round}

**Your Role**: If you are agent_2, you are the Leader with final decision authority. If you are agent_0 or agent_1, you are a Worker.

The task names you MUST use as JSON keys: {task_names}

== CONVERSATION HISTORY ==

<conversation_history>
{chat_history}
</conversation_history>

== YOUR PROPOSAL ==

<your_proposal>
{agent_proposal}
</your_proposal>

<your_explanation>
{agent_proposal_explanation}
</your_explanation>

== YOUR DECISION ON PARTNER'S PROPOSAL ==

<your_decision>
Accept: {accept_decision}
Reason: {decision_explanation}
</your_decision>

== COMPLETE INFORMATION FROM ALL AGENTS ==

**Agent 0 (Worker1) Efficiency values:**

<agent_0_efficiency_values>
{agent_0_efficiency}
</agent_0_efficiency_values>

**Agent 1 (Worker2) Efficiency values:**

<agent_1_efficiency_values>
{agent_1_efficiency}
</agent_1_efficiency_values>

**Agent 2 (Leader) Efficiency values:**

<agent_2_efficiency_values>
{agent_2_efficiency}
</agent_2_efficiency_values>

== TOTAL RESOURCES ==
**CRITICAL**: 
- Private resources stay SEPARATE per agent (do NOT add them together)
- Public resources MUST be ADDED: total_public = agent_0_obs + agent_1_obs + agent_2_obs

<total_private_resources>
Agent 0: {agent_0_private_resources}
Agent 1: {agent_1_private_resources}
Agent 2: {agent_2_private_resources}
</total_private_resources>

<total_public_resources>
{total_public_resources}
</total_public_resources>

== TASK REQUIREMENTS ==
<task_requirements>
{task_requirements}
</task_requirements>
(Note: Missing resource = 0 required)

Here is the optimal allocation based on total resources and overall preferences:
<best_allocation>
{best_allocation}
</best_allocation>

Note: the information above is based on your estimation so far. It may not be the ground truth. The tool's results are reliable only if your estimated resources and efficiency values are accurate.

== YOUR TASK (TALK PHASE) ==

Now, for the next round, please organize what to talk with your partners.
*Important*: You can say up to {max_character} characters, please carefully design your message.

Here are some suggestions (you are not required to follow all of them):
- Point out any constraint violations in proposals and explain why
- Propose corrections or improvements with explicit justification
- You may want to explain why you don't agree with your partner's proposal or state your agreement if the proposal is satisfactory.
- If you are the Leader, clearly indicate whether you are close to making a final decision

**Output Format**: Use XML tags. Output JSON directly (no markdown code blocks).

<reasoning_process>
(Your reasoning)
</reasoning_process>

<message_content>
(Your message - be concise, focus on key updates and your proposal)
</message_content>
'''