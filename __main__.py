import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--hello_world")
args = parser.parse_args()

if args.hello_world:
    print("hello world")