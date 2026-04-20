from . import TASK_REGISTRY

# i3l task prompts
TASKS = {
    "mug": "Pick up the mug and place it onto the plate.",
    "apple": "Pick up the apple and place it into the bowl.",
    "bowl": "Pick up the bowl and place it onto the plate.",
    "egg": "Pick up the egg and place it into the bowl.",
    "radio": "Pick up the radio on the table.",
}

# Register in global registry
TASK_REGISTRY["s2rg"] = TASKS