#!/usr/bin/env python
# -*- coding: UTF-8 -*-



TASK_DESCRIPTION = """
You are an AI agent, named {agent_name}. The user asked you to design a menu for treating their guests.
There are two guests in this task, {guest_name} and {partner_guest_name}, who are equally important. 

Note: 
This is a simulated game, you only need to maximize the reward values, which is decided by the guests' preferences. You don't need to consider too many aspects.
You must choose dishes according to the ingredients the *user* has in their home, the menu must be available to cook.
Each dish can only be chosen once, duplicated dishes will be neglected.
You observe all of the ingredients and all the guests' preferences. 

The preferences, at this stage, are numerical scores, so you can simply choose the dishes with the highest scores.

You need to determine the best menu based on the recipe, the guests' preferences, and the available ingredients.

"""

THINK_PROMPT_LEVEL_1_NO_TOOLS = TASK_DESCRIPTION + """

You will need to design the best menu, according to total ingredients and the overall preference.
Note that the "best menu" is the one that maximizes the sum of scores of all dishes in the menu, considering the total ingredients and the overall preference. 
This implies more dishes are better, however, the ingredients cost of the menu must be covered by the ingredients we have.

Here is the information you have so far:

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

Now you think and return the results, including your overall reasoning process, a final proposal (of menu) and a brief explanation.

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
