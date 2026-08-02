'''
Base Plugin Module for Zero2Neuro.

Provides the foundational `GenericPlugin` class that all custom plugins must inherit from,
as well as shared utilities for extracting data safely from the pipeline payload.
'''

import argparse

def require(kwargs, key):
    '''
    Extracts a required key from a kwargs payload.
    Raises a ValueError if the key is missing, ensuring plugins fail fast.
    '''
    if key not in kwargs:
        raise ValueError(f"Missing required plugin argument: '{key}'")
    return kwargs[key]


class GenericPlugin:
    '''
    Abstract base class for all Zero2Neuro plugins.
    
    Subclasses must:
      1. Set `self.role` in `__init__`.
      2. Override `call(**kwargs)` to provide their core logic.
    '''
    
    def __init__(self):
        # The execution stage where this plugin will fire (e.g., 'preprocess')
        self.role = None
        
        # Isolated parser to prevent flag collisions with the main framework
        self.parser = argparse.ArgumentParser(description='Generic Plugin')
        
        # Parsed namespace arguments (populated automatically by parse_args_inline)
        self.args = None
        
        # Optional string providing a usage example (shown in --list_plugins)
        self.example = None

    def parse_args_inline(self, inline_str):
        '''
        Parses plugin-specific configurations from a comma-separated string.
        Example: "method=minmax,verbose" -> parses as --method minmax --verbose.
        '''
        tokens = []
        
        if inline_str and inline_str.lower() != 'none':
            for pair in inline_str.split(','):
                pair = pair.strip()
                if not pair:
                    continue
                
                if '=' in pair:
                    key, val = pair.split('=', 1)
                    tokens.extend([f'--{key.strip()}', val.strip()])
                else:
                    tokens.append(f'--{pair}')
                    
        self.args = self.parser.parse_args(tokens)

    def call(self, **kwargs):
        '''
        Core execution logic for the plugin.
        Must be overridden by child classes.
        
        Returns:
            dict or None: Modified data to merge back into the main pipeline.
        '''
        pass
