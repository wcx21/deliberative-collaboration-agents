

TASK_DESCRIPTION = """
You are an AI agent, named {agent_name}. The user asked you to design a menu for treating their guests.
There are two guests in this task, {guest_name} and {partner_guest_name}, who are equally important. 

Note: 
This is a simulated game, you only need to maximize the reward values, which is decided by the guests' preferences. You don't need to consider too many aspects.
You must choose dishes according to the ingredients the *user* has in their home, the menu must be available to cook.
Each dish can only be chosen once, duplicated dishes will be neglected.
You observe all of the ingredients and all the guests' preferences. 

You need to determine the best menu based on the recipe, the guests' preferences, and the available ingredients.

To finish the task, we suggest you do the following steps in each round:
1. estimate the scores of the dishes on the recipe based on the information of the two guests.
2. think what is the best menu

"""

OBSERVE_PROMPT_LEVEL_2_TOOLS = TASK_DESCRIPTION + """
Here you will do the first step, you need to estimate  scores of the dishes.
Here is the information you have so far:

<total ingredients> 
{total_ingredients}
</total ingredients> 

<guest_1's information>
{person_1_info}
</guest_1's information>

<guest_2's information>
{person_2_info}
</guest_2's information>

<recipes> 
{recipes}
</recipes> 

You need to estimate the preferences of the dishes on the recipe based on the information of the two guests, with scores ranging from 0 to 10.

Now you output the results. You can include your thinking process into the output.
Your output must follow the format, no prefix or suffix.
The overall_preference must be in json format.
You must keep the tags with the angle brackets, but replace the content inside the tags.

Format:

<thinking_process>
thinking_process...
</thinking_process>

<overall_preference>
{{
    "dish_1":1.5,
    "dish_2":9.6,
   ...
}}
</overall_preference>

"""

THINK_PROMPT_LEVEL_2_TOOLS = TASK_DESCRIPTION + """

Here you will do the second step, you will need to design the best menu, according to your estimation on the overall preference.
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