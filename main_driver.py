"""
main driver
"""

import os
import sys
import time

import global_settings
import my_utility as mu

if __name__ == '__main__':
    TIME_I = time.time()

    FILE_DIR = os.path.abspath(os.path.join(os.path.realpath(
        sys.argv[0]), os.pardir, os.pardir, os.pardir))
    print(FILE_DIR)

    G_S = global_settings.get_setting(FILE_DIR)
