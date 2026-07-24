#!/usr/bin/env python
# -*- coding: UTF-8 -*-

## ========================= AGENT DESC =========================
'''
1 "Observe" agent, estimate the current state of the preference and the ingredients.
2 "Think" agent, propose a menu with a list of dishes based on the current state of the preference and the ingredients.
3 "Decision" agent, decide whether to accept the proposal or not.
4 "Talk" agent, communicate with another agent.
'''

## ======================== END DESC =========================


TASK_DESCRIPTION = """
You are an AI agent, named {agent_name}. The user asked you and another agent to design a menu for treating their guests.
There are two guests in this task, {guest_name} and {partner_guest_name}, who are equally important. 

Note: 
This is a simulated game, you only need to maximize the reward values, which is decided by the guests' preferences. You don't need to consider too many aspects.
You must choose dishes according to the ingredients the *user* has in their home, the menu must be available to cook.
You may observe a part of the ingredients and a part of the guests' preferences, while other agents may observe other parts. 

For each guest, you know some information about them, and you need to estimate the preferences for each dish based on the information of both guests. It is important to note that the agent you are working with may know some information that you do not.

You have to make a joint decision on the final menu with your partner, another AI agent. 
You two must reach an agreement in the end, otherwise you receive 0 reward.

Because you two can only observe partial information, you may need to communicate for collaboration.
You can communicate for {max_round} rounds. In each round, you can say up to {max_character} characters, messages that exceed this limit will be truncated.

To collaborate, we suggest you do the following steps in each round:
1. estimate the current state of the preference and the ingredients
2. think what is the best menu
3. decide whether to accept your partner's proposal or not
4. talk to your partner

"""

# Note that "total ingredients" = yours + your partner's, and "overall preference" = the average of yours and your partner's

OBSERVE_PROMPT = TASK_DESCRIPTION + """
Here you will do the first step, you need to estimate the "ingredients" and the "preference".
You estimate the information that your partner has observed (exclude yours), and the overall state will be automatically calculated.

Your initial observation:
<ingredients>
{current_obs}
</ingredients>

<guest_1's information>
{person_1_info}
</guest_1's information>

<guest_2's information>
{person_2_info}
</guest_2's information>

The recipes are:
<recipes> 
{recipes}
</recipes>

The dialog is:
<dialog>
{chat_history}
</dialog>

Note that you and your partner may observe different things, you must extract information from its utterance, and use that to update your estimation. 
Specifically, you need to do:
1. Estimate the preferences for each dish based on the information of {guest_name} and {partner_guest_name}, typically represented as a numerical value ranging from 0 to 10.
2. Estimate your partner's observation of <ingredients> and their preferences for the two guests.
2. Estimate the <total ingredients> and <overall preference>, based on your observation and your partner's. 

Note: Give neutral guess unless you have clear information. Your estimation will be used for planning in the next steps, incorrect estimation will lead to incorrect decision.

Now you output the results. You can include your thinking process into the output.
Your output must follow the format, no prefix or suffix.
The partner_ingredients, partner_preference, total_ingredients, and overall_preference must be in json format.
You must keep the tags with the angle brackets, but replace the content inside the tags.

Format:

<thinking_process_1>
thinking_process...
</thinking_process_1>

<partner_ingredients>
{{
    "ingredient_1":1,
    "ingredient_2":0,
   ...
}}
</partner_ingredients>

<partner_preference>
{{
    "dish_1":1.5,
    "dish_2":9.6,
   ...
}}
</partner_preference>

<thinking_process_2>
thinking_process...
</thinking_process_2>

<total_ingredients>
{{
    "ingredient_1":1,
    "ingredient_2":0,
   ...
}}
</total_ingredients>

<overall_preference>
{{
    "dish_1":1.5,
    "dish_2":9.6,
   ...
}}
</overall_preference>

"""

THINK_PROMPT = TASK_DESCRIPTION + """

Here you will do the second step, you will need to design the best menu, according to your estimation on the total ingredients and the overall preference.
Note that the "best menu" is the one that maximizes the sum of scores of all dishes in the menu, considering the total ingredients and the overall preference. 
This implies more dishes are better, however, the ingredients cost of the menu must be covered by the ingredients we have.

Note that you two share the same reward, which is the sum of the reward of each dish, and you will get 0 reward if you two cannot reach an agreement at the end.
Now is round {round_count} of the total {max_round} rounds.

Here is your current estimation:

<total ingredients> 
{total_ingredients}
</total ingredients> 

<overall preference> 
{overall_preference}
</overall preference> 

Here is the recipes that indicate the ingredients need by each dish:
<recipes> 
{recipes}
</recipes> 

Here is a list of dishes which already have sufficient ingredients.
<current available dishes> 
{available_dishes}
<current available dishes> 

Here is a list of missing ingredients for currently unavailable dishes.
<unavailable dishes> 
{unavailable_dishes_info}
<unavailable dishes> 

Here is the optimal dish list based on total ingredients and overall preference.
<best menu> 
{best_menu}
<best menu> 

Now you think and return the results, including your overall reasoning process, a final proposal (of menu) and a brief explanation.
Once your partner accept the menu proposal, the task will immediately end and you two get reward according to the menu in the proposal.
If you are not confident now, you can leave the menu proposal as a blank list, which will automatically be rejected.

Your output must follow the following format, no prefix or suffix.
The content inside the tags must be in json format.
You must keep the tags with the angle brackets, but replace the content inside the tags.

Format:
<reasoning_process>
XX
</reasoning_process>

<proposal>
{{
    "menu_proposal": ["dish_1","dish_2","dish_3","dish_4", ...],
    "explanation": "XX"
}}
</proposal>

"""

DECISION_PROMPT = TASK_DESCRIPTION + """

Here you will do the third step, base on your knowledge so far, you need to decide accept your partner's proposal or not.
Note that you two share the same reward, which is the sum of the reward of each dish, and you will get 0 reward if you two cannot reach an agreement at the end.

You have the following information:

Now is round {round_count} of the total {max_round} rounds.

Here is the recipes that indicate the ingredients need by each dish:
<recipes> 
{recipes}
</recipes> 

Your estimation of total ingredients and the overall preference:
<total ingredients> 
{total_ingredients}
</total ingredients> 

<overall preference> 
{overall_preference}
<overall preference> 

Based on your estimation:
Here is a list of dishes which already have sufficient ingredients.
<current available dishes> 
{available_dishes}
<current available dishes> 

Here is a list of missing ingredients for currently unavailable dishes.
<unavailable dishes> 
{unavailable_dishes_info}
<unavailable dishes> 

Here is the optimal dish list based on total ingredients and overall preference.
<best menu> 
{best_menu}
<best menu> 

Note: the information above is based on your estimation so far. It may not be the ground truth.

The dialog between you and your partner so far is:
<dialog>
{chat_history}
</dialog>

Your latest proposal is:
<your proposal>
{agent_proposal}
</your proposal>
Your explanation of your proposal is:
{agent_proposal_explanation}


The latest proposal from you partner is:
<partner's proposal>
{partner_proposal}
</partner's proposal>

<partner's explanation>
{partner_explanation}
</partner's explanation>

Now you need to decide whether to accept your partner's proposal or not.
Note that, once you decide to accept, the task will immediately end and you two get reward according to the menu in the proposal.


Before making a decision, you possibly may consider the following points:
1. When the menu is quite short and there are many round remaining, you may still have time for seeking for a better outcome. 
2. When there are few remaining dialogue rounds, aim to reach an agreement as soon as possible.
3. Your observation is partial, the available ingredients are the sum of you have observed and your partner has observed.
4. A non-perfect menu will still receive partial reward.


Your output must follow the following format, no prefix or suffix.
The content inside the tags must be in json format.
You must keep the tags with the angle brackets, but replace the content inside the tags.

Format:
<reasoning_process>
XX
</reasoning_process>

<decision>
{{
    "accept_decision": true/false,
    "explanation": "XX"
}}
</decision>
"""

TALK_PROMPT = TASK_DESCRIPTION + """

Here you will do the four step, also the final step in this round.
You have made an estimation, a proposal, and you have not reach an agreement with your partner so far.
Therefore, you must have something to talk with your partner.

Here is all the information so far:

Now is round {round_count} of the total {max_round} rounds.

Here is the recipes that indicate the ingredients need by each dish:
<recipes> 
{recipes}
</recipes> 

Your initial observation:
<ingredients>
{current_obs}
</ingredients>

<guest_1's information>
{person_1_info}
</guest_1's information>

<guest_2's information>
{person_2_info}
</guest_2's information>

The dialog is:
<dialog>
{chat_history}
</dialog>

Your estimation of your partner's observation and the total ingredients and the overall preference:

<estimated partner's ingredients> 
{partner_ingredients}
</estimated partner's ingredients> 

<estimated partner's preference> 
{partner_preference}
</estimated partner's preference> 

<estimated total ingredients> 
{total_ingredients}
</estimated total ingredients> 

<estimated overall preference> 
{overall_preference}
<estimated overall preference> 

Base on your estimation:
Here is a list of dishes which already have sufficient ingredients.
<current available dishes> 
{available_dishes}
<current available dishes> 

Here is a list of missing ingredients for currently unavailable dishes.
<unavailable dishes> 
{unavailable_dishes_info}
<unavailable dishes> 

Here is the optimal dish list based on total ingredients and overall preference.
<best menu> 
{best_menu}
<best menu> 
Note: the information above is based on your estimation so far. It may not be the ground truth.

Base on these, your latest proposal is: 
<your proposal>
{agent_proposal}
</your proposal>

The latest proposal from you partner is:
<partner's proposal>
{partner_proposal}
</partner's proposal>

You rejected your partner's proposal, because:
<decision explanation>
{decision_explanation}
</decision explanation>

Now, for the next round, please organize what to talk with your partner.
*Important*: You can say up to {max_character} characters, please carefully design your message.

Here are some suggestions (you are not required to follow all of them):
- It is very important to share your observation, including ingredients and preferences.
- Especially you should share the ingredients you have and those you are still waiting for.
- You may want to explain why you don't agree with your partner's proposal 

Your output must follow the following format, no prefix or suffix.
The content inside the tags must be in json format.
You must keep the tags with the angle brackets, but replace the content inside the tags.

Format:
<reasoning_process>
XX
</reasoning_process>

<message_content>
(The words you want to say to your partner)
</message_content>

"""
