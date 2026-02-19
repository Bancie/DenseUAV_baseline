"""Checkpoint cleanup utility for the DenseUAV baseline.

Scans all run subdirectories under ``../model/`` and removes early-epoch
checkpoint files — specifically, any ``net_*.pth`` file whose epoch number
(the digit at index 4 of the filename) is less than 1.  Files whose name has
``'a'`` at index 5 are skipped unconditionally.

This is useful for pruning storage-heavy checkpoints from aborted or
exploratory runs while retaining only later, higher-quality snapshots.

Example:
    Run from the ``baseline/tool/`` directory::

        python clear_model.py

    Or from the ``baseline/`` directory::

        python tool/clear_model.py

Outputs:
    Prints the path of each deleted checkpoint to stdout and removes the
    file from disk.
"""

import os

root = '../model/'
nn = []
for f in os.listdir(root):
    if not os.path.isdir(root+f):
        continue
    for ff in os.listdir(root+f):
        if ff[0:3] == 'net':
            if ff[5] =='a':
                continue
            if int(ff[4])<1:
                path = root+f+'/'+ff
                print(path)
                os.remove(path)
