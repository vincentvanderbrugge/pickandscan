#!/bin/bash

python -m pdb eval_all.py \
--model /local/home/vincentv/code/motion_segment2/checkpoints/xmem_saves/XMem.pth \
--output /local/home/vincentv/code/motion_segment2/data/xmem_output \
--dataset G \
--generic_path /local/home/vincentv/code/motion_segment2/data/dayfour2/process_dir/xmem/xmem_input_dayfour2