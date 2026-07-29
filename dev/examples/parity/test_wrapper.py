import sys
import os
import traceback

sys.path.insert(0, os.path.abspath('../../src'))

try:
    import runpy
    sys.argv = ['../../src/zero2neuro.py', '@data_config.txt', '@network_config.txt', '@experiment_config.txt', '--nogo', '-v']
    runpy.run_path('../../src/zero2neuro.py', run_name='__main__')
except Exception as e:
    print("CAUGHT EXCEPTION:")
    traceback.print_exc(file=sys.stdout)
    sys.stdout.flush()
    sys.exit(1)
except SystemExit as e:
    print(f"SYSTEM EXIT with code {e.code}")
    sys.stdout.flush()
    sys.exit(e.code)
