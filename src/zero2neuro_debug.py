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
                print()
                print(msg)
                sys.exit()
            case 1: 
                # Just show error message and location of call
                print()
                frame = inspect.currentframe().f_back
                line_num = frame.f_lineno
                file_name = inspect.stack()[1].filename
                print(msg, "Occured at line:", line_num, "in:", file_name)
                sys.exit()
            case _:
                # Show full stack trace 
                stack = traceback.extract_stack()
                print()
                print("Stack:")
                for f in stack:
                    print(f)
                #print(stack)
                print()
                print(msg)
                sys.exit()
