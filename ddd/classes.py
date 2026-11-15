"""Class map of StateFarm. Source: docs/PLAN.md section 3.4.

Own recordings use the names. Their labels map to the StateFarm codes by name.
"""

# (name, StateFarm code, status)
CLASSES = [
    ("safe_driving", "c0", "safe"),
    ("texting_right", "c1", "distracted"),
    ("phone_right", "c2", "distracted"),
    ("texting_left", "c3", "distracted"),
    ("phone_left", "c4", "distracted"),
    ("radio", "c5", "distracted"),
    ("drinking", "c6", "distracted"),
    ("reaching_behind", "c7", "distracted"),
    ("hair_makeup", "c8", "distracted"),
    ("talking_passenger", "c9", "distracted"),
]

NAMES = [c[0] for c in CLASSES]
STATEFARM_TO_NAME = {c[1]: c[0] for c in CLASSES}
NAME_TO_STATEFARM = {c[0]: c[1] for c in CLASSES}
STATUS = {c[0]: c[2] for c in CLASSES}
