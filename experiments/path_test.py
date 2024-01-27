import sys
import os
import pkg_resources
from pprint import pprint
import sys
# sys.path.extend(['/local/home/vincentv/anaconda3/envs/torchtest2/bin',
#                          '/local/home/vincentv/anaconda3/condabin',
#                          '/usr/local/cuda-11/bin',
#                          '/usr/local/cuda-11/bin',
#                          '/usr/local/sbin',
#                          '/usr/local/bin',
#                          '/usr/sbin',
#                          '/usr/bin',
#                          '/sbin',
#                          '/bin',
#                          '/usr/games',
#                          '/usr/local/games',
#                          '/snap/bin'])


# os.environ["PATH"] += os.pathsep + "/local/home/vincentv/anaconda3/envs/torchtest2/bin:/local/home/vincentv/anaconda3/condabin:/usr/local/cuda-11/bin:/usr/local/cuda-11/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"



env = {n: (v.split(os.pathsep) if 'PATH' in n else v) for n, v in os.environ.items()}
pprint({
    'sys.version_info': sys.version_info,
    'sys.prefix': sys.prefix,
    'sys.path': sys.path,
    'pkg_resources.working_set': list(pkg_resources.working_set),
    'os.environ': env,
})