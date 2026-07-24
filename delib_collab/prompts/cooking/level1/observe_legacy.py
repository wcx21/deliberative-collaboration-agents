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
You are an AI agent. The user asked you and another agent to design a menu for treating their guests.

There are two guests in this task, Alice and Bob, they are equally important. 

This is a simplified task, where you must choose dishes according to the ingredients the *user* has in their home and the guests' preferences.

Because you are an AI agent, you may observe a part of the ingredients and a part of the guests' preferences, while other agents may observe other parts. 

The preferences, at this stage, are numerical scores, so you can simply choose the dishes with the highest scores.

You have to make a joint decision on the final menu with your partner, another AI agent. 
You two must reach an agreement in the end, otherwise you receive 0 reward.

Because you two can only observe partial information, you may need to communicate for collaboration.
You can communicate for {max_round} rounds. In each round, you can say up to {max_character} characters, messages that exceed this limit will be truncated.

To collaborate, we suggest you to do the following steps in each round:
1. estimate the current state of the preference and the ingredients
2. think what is the best menu
3. decide whether to accept your partner's proposal or not
4. Talk to your partner

"""


OBSERVE_PROMPT = TASK_DESCRIPTION + """
Here you will do the first step, you need to estimate the "total ingredients" and the "overall preference", according to your initial observation and the dialog between you and your partner. 
Note that "total ingredients" = yours + your partner's, and "overall preference" = the average of yours and your partner's

Your initial observation:
<ingredients>
{current_obs}
</ingredients>

<{guest_name}'s preference>
{score_dict}
</{guest_name}'s preference>

The dialog is:
<dialog>
{chat_history}
</dialog>

Note that you and your partner may observe different things, you must extract information from its utterance, and use that to update your estimation. 
Specifically, you need to do:
1. Estimate your partner's observation of <ingredients> and <{partner_guest_name}'s preference> 
2. Estimate the <total ingredients> and <overall preference>, based on your observation and your partner's. 

Now you output the results. You can include your thinking process into the output.

(a brief thinking process to for step 1)

<partner's ingredients> 
(a json dict akin to your observation of <ingredients>)
</partner's ingredients> 

<{partner_guest_name}'s preference> 
(a json dict akin to your observation of <{guest_name}'s preference>)
<{partner_guest_name}'s preference> 
 
(a brief thinking process to for step 2)
 
<total ingredients> 
(a json dict akin to your observation of <ingredients>)
</total ingredients> 

<overall preference> 
(a json dict akin to your observation of <{guest_name}'s preference>)
<overall preference> 
 
"""
