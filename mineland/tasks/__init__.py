import importlib

from omegaconf import OmegaConf

from .utils import *

from .base_task import BaseTask
from .amongus_task import AmongUsTask

# ===== Main Make =====

def make(**kwargs):
    '''Make a task environment.

    Args:
        task_id (str): The id of the task. "playground" for a bare sandbox,
            or any key defined in `description_files/amongus_tasks.yaml`.

    Example:
        >>> env = mineland.make("playground", agents_count=1, agents_config=[{"name": "MineflayerBot0"}])
        >>> env = mineland.make("amongus_2_imposter_6_crewmates", agents_count=8)
    '''

    if 'agents_count' not in kwargs:
        raise ValueError("agents_count must be provided in the arguments.")

    if 'server_host' in kwargs or 'server_port' in kwargs:
        raise ValueError("server_host and server_port should not be provided in the Task Mode!\nBecause Benchmark only works in the local environment.")

    if 'task_id' not in kwargs:
        raise ValueError("task_id must be provided in the arguments.")

    def add_mode_argument():
        if 'mode' not in kwargs or (kwargs['mode'] not in ['cooperative', 'competitive', 'deceptive']):
            kwargs['mode'] = 'cooperative'
            print("task mode has been modified as cooperative")

    if 'agents_config' not in kwargs:
        kwargs['agents_config'] = [{'name': f'MineflayerBot{i}'} for i in range(kwargs['agents_count'])]

    if kwargs['task_id'].startswith("playground"):
        env = _make_playground(**kwargs)
    elif kwargs['task_id'].startswith("amongus"):
        add_mode_argument()
        env = _make_amongus(**kwargs)
    else:
        raise ValueError(f"Invalid task_id: {kwargs['task_id']}")
    return env

# ===== Load Datas =====

def _resource_file_path(fname) -> str:
    with importlib.resources.path("mineland.tasks.description_files", fname) as p:
        return str(p)

# load among us tasks
AMONGUS_TASKS = OmegaConf.load(
    _resource_file_path("amongus_tasks.yaml")
)
# check no duplicates
assert len(set(AMONGUS_TASKS.keys())) == len(AMONGUS_TASKS)

# ===== Playground =====

def _make_playground(**kwargs):
    env = BaseTask(**kwargs)
    return env

# ===== AmongUs =====

def _make_amongus(**kwargs):
    task_id = kwargs['task_id']
    if task_id not in AMONGUS_TASKS:
        raise ValueError(f"Invalid task_id: {task_id}")

    task = AMONGUS_TASKS[task_id]
    goal = task["goal"]
    guidance = task["guidance"]

    env = AmongUsTask(
        goal=goal,
        guidance=guidance,
        **kwargs,
    )
    return env
