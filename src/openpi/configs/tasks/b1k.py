from . import TASK_REGISTRY

# i3l task prompts
TASKS = {
    "open": "Open the door of the fridge.",
    "radio": "Pick up the radio on the table.",
    "open_radio": "Open the fridge door, then pick up the radios.",
    "books": "Place the basket onto the table, and put the two books from the shelf into the basket.",
    "popcorn": "Pick up the red popcorn bag from the shelf and place it into the microwave.",
    "bread": "Retrieve the bowl from the drawer, put the two bread in from the plate, and place the bowl on the right side of the countertop.",
}

# Register in global registry
TASK_REGISTRY["b1k"] = TASKS
