'''
Plugin Manager for Zero2Neuro.

Handles dynamic discovery, loading, and execution routing of Python plugins.
It allows the core framework to be extended without modifying internal dependencies.
'''

import os
import sys
import importlib.util
from zero2neuro_debug import print_debug

class PluginManager:
    '''
    Manages discovery, loading, and execution of Zero2Neuro plugins.
    '''
    def __init__(self, plugin_paths):
        self.plugin_paths = [os.path.abspath(p) for p in plugin_paths]
        default_dir = os.path.join(os.path.dirname(__file__), 'plugins')
        if default_dir not in self.plugin_paths:
            self.plugin_paths.append(default_dir)
        self.plugins = []

    def _find_plugin_file(self, name):
        '''Resolves a plugin name or explicit path to an absolute .py file path.'''
        if name.endswith('.py') or os.sep in name or '/' in name:
            candidate = os.path.abspath(name)
            if os.path.exists(candidate):
                return candidate
            raise FileNotFoundError(f"Plugin file not found: {candidate}")
            
        for path in self.plugin_paths:
            candidate = os.path.join(path, f"{name}.py")
            if os.path.exists(candidate):
                return candidate
                
        raise FileNotFoundError(f"Plugin '{name}' not found in paths: {self.plugin_paths}")

    def _import_module(self, name, file_path):
        '''Dynamically imports a Python file as a module.'''
        spec = importlib.util.spec_from_file_location(name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def _get_plugin_class(self, module, name):
        '''Extracts the target plugin class from the module.'''
        if not hasattr(module, name):
            raise AttributeError(f"Plugin file '{module.__file__}' must contain class '{name}'")
        return getattr(module, name)

    def load_plugins(self, plugin_list, debug_level=0):
        '''Discovers, instantiates, and configures plugins based on CLI arguments.'''
        if not plugin_list:
            return
            
        for item in plugin_list:
            if ',' in item:
                name_role, arg_str = item.split(',', 1)
                name_role = name_role.strip()
                arg_str = arg_str.strip()
            else:
                name_role = item.strip()
                arg_str = 'none'
            
            name, role = name_role.rsplit('-', 1) if '-' in name_role else (name_role, None)
            
            file_path = self._find_plugin_file(name)
            class_name = os.path.splitext(os.path.basename(file_path))[0]
            
            module = self._import_module(class_name, file_path)
            plugin_class = self._get_plugin_class(module, class_name)
            plugin = plugin_class()
            
            if role:
                plugin.role = role
                
            plugin.parse_args_inline(arg_str)
            self.plugins.append(plugin)
            print_debug(f"Loaded plugin '{class_name}' with role '{plugin.role}'", 1, debug_level)

    def apply_plugins(self, role, debug_level=0, **kwargs):
        '''Executes all registered plugins matching the specified execution role.'''
        for plugin in self.plugins:
            if plugin.role == role:
                result = plugin.call(**kwargs)
                if result:
                    kwargs.update(result)
        return kwargs

def list_plugins(plugin_paths):
    '''Scans configured plugin directories and prints a registry of available plugins.'''
    default_dir = os.path.join(os.path.dirname(__file__), 'plugins')
    if default_dir not in plugin_paths:
        plugin_paths.append(default_dir)

    print("\n--- Available Plugins ---")
    for path in plugin_paths:
        if not os.path.exists(path):
            continue
            
        files = sorted(f for f in os.listdir(path) if f.endswith('.py') and not f.startswith('__'))
        for p_file in files:
            name = os.path.splitext(p_file)[0]
            file_path = os.path.join(path, p_file)
            
            spec = importlib.util.spec_from_file_location(name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            if hasattr(module, name):
                plugin = getattr(module, name)()
                
                print(f"\nPlugin: {name}")
                print(f"  |   Description : {plugin.parser.description or 'N/A'}")
                print(f"  |   Role        : {getattr(plugin, 'role', 'None')}")
                print(f"  |   Arguments   :")
                for action in plugin.parser._actions:
                    if action.dest != 'help':
                        flag = action.option_strings[0] if action.option_strings else action.dest
                        print(f"  |       {flag:30s}  {action.help}")
                        
                if getattr(plugin, 'example', None):
                    print(f"  |   Example     : {plugin.example}")
