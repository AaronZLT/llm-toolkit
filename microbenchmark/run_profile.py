import json
import argparse
import numpy as np

from profiler import *


def add_parser_arg(parser):
    parser.add_argument('--trace_path', type=str,
                    help='The path of trace',
                    default="./bs32_recomp.json")
    parser.add_argument('--config_path', type=str,
                    help="The path of config file",
                    default="./recomputation_config.json")

def get_raw_trace(trace_path):
    with open(trace_path, "r") as json_file:
        raw_trace = json.load(json_file)
    json_file.close()
    return raw_trace

def get_config(config_path):
    with open(config_path, "r") as json_file:
        config = json.load(json_file)
    json_file.close()
    return config
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_parser_arg(parser)
    args = parser.parse_args()

    raw_data = get_raw_trace(args.trace_path)
    config = get_config(args.config_path)
    tracer = BenchTracer(config, raw_data)
    tracer.extract()