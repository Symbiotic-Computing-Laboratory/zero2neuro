'''
Generic debugging tools
'''

def print_debug(threshold:int, debug_level:int, strg:str):
    if debug_level >= threshold:
        print("#############", threshold)
        print(strg)
        print("#############")

