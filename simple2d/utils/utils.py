import json
import pathlib
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
from copy import copy


def save_config(
        config: dict = {},
        timestamp: Optional[str] = None,
        folder: os.PathLike = pathlib.Path(__file__).parent.parent.parent
        .joinpath('configs'),
        extras: dict = {}
        ) -> None:
    """Saves a json of the config dictionary to a timestamped model folder."""
    to_save = copy(config)
    if extras:
        to_save.update(extras)
    timestamp = timestamp or get_time_str()
    if not os.path.exists(folder):
        os.makedirs(folder)
    file = os.path.join(folder, 'run_'+timestamp+'.json')
    with open(file, 'w') as fp:
        json.dump(to_save, fp)
    print(f'Config saved to: {str(file)}')


def load_config(path: os.PathLike) -> dict:
    """Loads a json as config dictionary from path"""
    with open(path, 'r') as fp:
        d = json.load(fp)
    return d


def get_time_str():
    """returns string of current EST time as %Y%m%d-%H%M%S"""
    timezone_offset = -5.0  # Eastern Standard Time (UTC−05:00)
    tzinfo = timezone(timedelta(hours=timezone_offset))
    now = datetime.now(tzinfo)
    now_str = now.strftime('%Y%m%d-%H%M%S')
    return now_str
