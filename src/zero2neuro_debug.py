'''
Generic debugging tools
'''
import inspect
import traceback
import sys

def print_debug(threshold:int, debug_level:int, strg:str):
    if debug_level >= threshold:
        print("#############", threshold)
        print(strg)
        print("#############")

# TODO: Figure out what a debug_lvl of 0 does, make a format for what an error message should be.
def handle_error(msg: str, debug_lvl: int):
        match debug_lvl:
            case 0:
                # Placeholder, do nothing?
                print("placeholder")
            case 1: 
                # Just show error message and location of call
                frame = inspect.currentframe().f_back
                line_num = frame.f_lineno
                print(msg, "Occured at line:", line_num)
                sys.exit()
            case 2:
                # Show full stack trace 
                stack = traceback.extract_stack() 
                print(stack)
                sys.exit()