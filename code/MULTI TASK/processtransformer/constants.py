import enum

@enum.unique
class Task(enum.Enum):
  """Look up for tasks."""
  
  NEXT_ACTIVITY = "next_activity"
  TIMES = "times"

