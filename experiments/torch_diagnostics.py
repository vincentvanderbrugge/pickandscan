import sys
import os
import pkg_resources
from pprint import pprint

env = {n: (v.split(os.pathsep) if 'PATH' in n else v) for n, v in os.environ.items()}
pprint({
    'sys.version_info': sys.version_info,
    'sys.prefix': sys.prefix,
    'sys.path': sys.path,
    'pkg_resources.working_set': list(pkg_resources.working_set),
    'os.environ': env,
})