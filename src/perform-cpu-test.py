
import re
from pathlib import Path

instructions = [
    'atomicrmw',
    'extractelement', 'insertelement',
    'shufflevector'
]

prefix = '//inst_stats:'
srcstats = re.compile('([a-z0-9\.]+)\(([0-9]+)\),?')

if __name__ == "__main__":
    import sys
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--src-file", type=Path, required=True)
    parser.add_argument("--bin-file", type=Path, required=True)
    parser.add_argument("--llvm-file", type=Path, required=True)
    args = parser.parse_args()

    print("Perform test case", args.src_file, "...", "", end="")

    # print("executing", args.bin_file)
    result = subprocess.run(args.bin_file, capture_output=True, encoding='utf-8')
    if result.returncode != 0:
        print('failed with incorrect computation!')
        print(result.stdout)
        sys.exit(result.returncode)

    inst_stats = dict([(inst, 0) for inst in instructions])
    found_inst_stats = False

    with open(args.src_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().replace(' ', '').replace('\t', '')
            if not line.startswith(prefix):
                continue

            line = line[len(prefix):]
            # print("Found inst_stats:", line)
            found_inst_stats = True

            for m in srcstats.finditer(line):
                inst_stats[m.group(1)] = int(m.group(2))

    if not found_inst_stats:
        sys.exit(0)

    # print()
    # print("inst_stats:", inst_stats)

    current_stats = {}
    for key in inst_stats.keys():
        current_stats[key] = 0

    with open(args.llvm_file, 'r', encoding='utf-8') as file:
        for line in file:
            if line.lstrip().startswith('declare'):
                continue

            for key in inst_stats.keys():
                if key not in line:
                    continue

                current_stats[key] += 1

    # print("current_stats:", current_stats)

    passed = True
    for key in inst_stats.keys():
        passed = current_stats[key] <= inst_stats[key]
        if not passed:
            break

    if passed:
        print('passed!')
    else:
        print('failed with wrong instructions!')

    for key in inst_stats.keys():
        if current_stats[key] != inst_stats[key]:
            print('(!)', end=' ')
        else:
            print('   ', end=' ')
        print(key, '(', inst_stats[key], ') ->', current_stats[key])

    if not passed:
        sys.exit(-1)
