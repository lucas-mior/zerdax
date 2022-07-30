#!/usr/bin/python

import argparse
from algo import full
from termcolor import colored, cprint

def parseargs():
    parser = argparse.ArgumentParser(description='Convert chess photo to FEN')

    parser.add_argument('image', type=str, default=None, help='Image filename')

    args = parser.parse_args()
    return args

def Main():
    args = parseargs()
    image = args.image
    cout = colored("============ zerdax {} ============".format(image), "white", attrs=['bold'])
    print(cout)
    fen = full(image)
    cout = colored("FEN:", "white", attrs=['bold'])
    print(cout, fen)

if __name__ == '__main__':
    Main()
