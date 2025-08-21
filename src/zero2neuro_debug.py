'''
Generic debugging tools
'''
import inspect
import traceback
import sys

def print_debug(strg:str="", threshold:int=0, debug_level:int=1):
    '''
    Conditionally generate debugging output

    :param strg: String to print out
    :param threshold: Fixed threshold at which the output is generated
    :param debug_level: The current level of debugging.  Typically args.debug
    
    '''
    
    if debug_level >= threshold:
        print("#############", threshold)
        print(strg)
        print("#############")

def handle_error(msg: str="", verbosity: int=2):
    '''
    Generate a terminal error message and exit python
    
    :param msg: Message to print
    :param verbosity: Level of output to generate.
                   0 = message only,
                   1 = report file and line number,
                   2 = report full stack trace

                   This is typically args.verbose
    
    '''
    match verbosity:
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
