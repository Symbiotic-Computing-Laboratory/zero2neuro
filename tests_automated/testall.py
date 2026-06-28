'''
Loop over all local .sh files & execute each.

If one ends in error, then stderr is printed out and scanning halts.

Usage:

pytest testall.py

'''


import subprocess
from pathlib import Path
import pytest

scripts = sorted((Path(__file__).parent).glob("*.sh"))

@pytest.mark.parametrize("script", scripts, ids=lambda p: p.name)
def test_script(script):
    '''
    Test a single script
    '''
    
    r = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    
    assert r.returncode == 0, f"{script.name} failed:\n{r.stderr}"
    
