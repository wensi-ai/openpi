from . import TASK_REGISTRY

# i3l task prompts
TASKS = {
    "pnp": "Pick up and place the strawberry onto the white plate.",
    "books": "Place the basket onto the table, and put the two books from the shelf into the basket."
}

# Register in global registry
TASK_REGISTRY["i3l"] = TASKS
