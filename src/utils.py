# ---------------------------------------------------------
# Utils Implementation
# Licensed under The MIT License [see LICENSE for details]
# Written by Cheng-Bin Jin
# Email: sbkim0407@gmail.com
# ---------------------------------------------------------

import os
import logging
import tensorflow as tf
import sys

def init_logger(logger, logDir, name, isTrain):
    logger.propagate = False  # solve print log multiple times problem
    fileHandler, streamHandler = None, None

    if isTrain:
        formatter = logging.Formatter(' - %(message)s')

        # File handler
        fileHandler = logging.FileHandler(os.path.join(logDir, name + '.log'))
        fileHandler.setFormatter(formatter)
        fileHandler.setLevel(logging.INFO)

        # Stream handler
        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)

        # Add handlers
        if not logger.handlers:
            logger.addHandler(fileHandler)
            logger.addHandler(streamHandler)

    return logger, fileHandler, streamHandler

def print_metrics(itr, kargs):
    print("*** Iteration {}  ====> ".format(itr))
    for name, value in kargs.items():
        print("{} : {}, ".format(name, value))
    print("")
    sys.stdout.flush()