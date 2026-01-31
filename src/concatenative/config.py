import tomllib
import collections.abc
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    'segmentation': {
        'max_duration_s': 0.2,
        'onset': {
            'hop_length': 512,
            'backtrack': False,
            'normalise': False,
        }
    }, 
    'features': {
        'frame_length': 2048,
        'hop_length': 512, 
        'pitch': {
            'fmin': 50,
            'fmax': 5000
        }
    }
}

def deep_merge(dest: dict, target: dict) -> dict:
    '''
    Recursively merges 'target' into 'dest'
    If a key in target exists in dest and they are both dictionaries it merges them
    Otherwise the value from target overrides dest
    '''

    for k, v in target.items():
        if isinstance(v, collections.abc.Mapping):
            dest[k] = deep_merge(dest.get(k, {}), v)
        else:
            dest[k] = v

    return dest


def load_config(config_path: Path | None = None) -> dict:

    final_config = deep_merge({}, DEFAULT_CONFIG)

    if not config_path:
        logger.info("No config file provided. Using default settings.")
        return final_config
    
    if not config_path.exists():
        logger.warning(f"Config file {config_path.resolve()} not found. Using default settings")
        return final_config

    logger.info(f"Loading config from: {config_path.resolve()}")
    try:
        with config_path.open('rb') as f:
            user_config = tomllib.load(f)
            final_config = deep_merge(final_config, user_config)
    
    except tomllib.TOMLDecodeError as e:
        logger.error(f"Could not parse config file: {config_path.resolve()}")
        logger.info(f"Falling back to default config settings")
        final_config = deep_merge({}, DEFAULT_CONFIG)

    return final_config

