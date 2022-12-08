# Tool to run cacti with a range of configurations to compare results.
# Saji Champlin 2022
# Written for EE5204

import subprocess
import csv
import itertools
from math import log2
import numpy as np
import os
from pathlib import Path
import multiprocessing
import argparse
# this is the part of the file that actually matters - it's what we change.
# This is combined with the defaults file (separate because large) to
# form a valid cacti config.
template_cfg = """
-size (bytes) {cache_size}
-block size (bytes) {block_size}
-associativity {associativity}
-tag size (b) {tag_size}
-search port {search_port}
"""
# size = TLB block size * number of entries
# VAS - virtual address space
# tag size = VPN - log2(num_entries) i.e 256 elements in TLB, use lower 8 bits
# of VPN to index into cache, use remaining bit as tag to check.
# TLB block size = tag size + misc bits (4)
# associativity ???
# params we pick: VAS, page size, associativity, num_entries (8-2048)
# params we calc: size, tag size, block size.
# additional measurements - delay, area, power

# tag_sizes = [12,14,16,20,24,28,32,36]
addr_size = [48, 64]
page_size = [12]
associativity = [0, 1, 2, 4, 8, 16, 32, 64]

num_entries = np.arange(8, 4096, 64)

with open('cacti_base.cfg', 'r') as f:
    base_config = f.read()


# we want a way to run the cacti suite.
def run_cacti(params):
    # run a cacti simulation and return the results in a dictionary.
    # params is a dictionary of params to be passed to the template
    print(f"running cacti with params: {params}")

    if multiprocessing.parent_process():  # this is none if we are main proc
        config_name = f'paramfile_{multiprocessing.current_process().name}.cfg'
    else:
        config_name = "paramfile.cfg"
    with open(config_name, 'w') as tf:
        # we need to use CRLF since cacti can't parse normal files.
        tf.write(template_cfg.format(**params).replace('\n', '\r\n'))
        tf.write(base_config.replace('\n', '\r\n'))

    # run cacti and get the output
    ex = subprocess.run(['./cacti', '-infile', config_name],
                        capture_output=True)

    if (ex.returncode != 0):
        print(ex.stderr.decode())
    ex.check_returncode()

    output_filename = config_name + ".out"
    with open(output_filename, 'r') as f:
        p_reader = csv.DictReader(f)
        data = list(p_reader)[-1] | params

    # cleanup the things

    os.remove(config_name)
    os.remove(output_filename)

    return data


# run_cacti({'cache_size': 4096, 'block_size': 64, 'associativity': 0, 'tag_size': 11})
tests = itertools.product(addr_size, page_size,
                          associativity, num_entries)


# run the tests
def run_test(test):
    addr_size = test[0]  # bits
    page_size = test[1]  # bits
    assoc = test[2]      # num
    n_entries = test[3]  # count

    print(f"running test for {addr_size} {page_size} {assoc} {n_entries}")

    vpn_size = addr_size - page_size

    index_size = 0
    if (assoc != 0):
        # number of bits needed for indexing.
        index_size = int(log2(assoc))

    # tag size is VPN - number of bits to index all tlb entries
    tag_size = vpn_size - int(log2(n_entries)) - index_size
    # the size of the entry in the TLB
    # the 2 is for status (dirty and valid bits)
    block_size = tag_size + index_size + vpn_size + 2

    cache_size = block_size * n_entries
    print(f"caclulated values: vpn={vpn_size} tag={tag_size} block={block_size} cache={cache_size}")
    # log2(n_entries) - log2(assoc) > 3 for it to work.
    # so we skip this run if it isn't going to happen.
    if log2(n_entries) - (0 if assoc == 0 else log2(assoc)) <= 4:
        # don't go!
        print("we're chillin cause associativity is too high!")
        return {}
    else:
        print("we're going to run!")
        return run_cacti({'cache_size': cache_size, 'block_size': block_size,
                          'associativity': assoc, 'tag_size': tag_size,
                          'search_port': 1 if assoc == 0 else 0,
                          'n_entries': n_entries, 'addr_size': addr_size,
                          'page_size': page_size})


if __name__ == "__main__":
    # gather arguments and run tests.
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", help="Where to output the CSV data",
                        default="output.csv", type=Path)
    parser.add_argument("--mp", help="Number of cores to use, 0 means all \
                        cores,-1 means single-threaded",
                        default=-1, type=int)
    a = parser.parse_args()

    if a.mp != -1:
        print("Using multicore mode...")
        with multiprocessing.Pool(a.mp if a.mp else None) as p:
            res = list(filter(None, p.map(run_test, tests)))
    else:
        print("using single core mode")
        res = list(filter(None, map(run_test, tests)))

    print(f"finished running tests. saving data to {a.output}")
    with open(a.output, 'w+') as out:
        d_writer = csv.DictWriter(out, fieldnames=res[0].keys())
        d_writer.writeheader()
        for row in res:
            d_writer.writerow(row)
    print("finished!")
