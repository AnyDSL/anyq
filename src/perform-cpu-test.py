
import re
from pathlib import Path

instructions = [
    'atomicrmw',
    'extractelement', 'insertelement',
    'shufflevector'
]

prefix = re.compile('^//inst_stats\[([a-z0-9_\-]+)\]:')
srcstats = re.compile('([a-z0-9\._]+)\(([0-9]+)\),?')

if __name__ == "__main__":
    import sys
    import argparse
    import subprocess

    parser = argparse.ArgumentParser()
    parser.add_argument("--src-file", type=Path, required=True)
    parser.add_argument("--bin-file", type=Path, required=True)
    parser.add_argument("--llvm-file", type=Path, required=True)
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    # print(Path.cwd())
    print("Perform test case", args.src_file, "...", "", end="")

    # print("executing", args.bin_file)
    result = subprocess.run(str(args.bin_file.resolve()), capture_output=True, encoding='utf-8')
    if result.returncode != 0:
        print('failed with incorrect computation!')
        print(result.stdout)
        sys.exit(result.returncode)

    inst_stats = dict()
    found_inst_stats = False

    with open(args.src_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip().replace(' ', '').replace('\t', '')

            m = prefix.match(line)
            if m is None:
                continue

            isa = m.group(1)
            inst_stats[isa] = dict([(inst, 0) for inst in instructions])

            remaining_line = line[len(m.group(0)):]
            # print("Found inst_stats:", remaining_line)
            # print("for isa", isa)
            found_inst_stats = True

            for m in srcstats.finditer(remaining_line):
                inst_stats[isa][m.group(1)] = int(m.group(2))

    if not found_inst_stats:
        sys.exit(0)

    # print()
    # print("inst_stats:", inst_stats)

    llvm_file = args.llvm_file
    if not args.llvm_file.is_file():
        llvm_file = args.llvm_file.parent / args.config / args.llvm_file.name

    if not llvm_file.is_file():
        print("llvm file not found:", args.llvm_file)
        sys.exit(-2)

    current_isa = None
    current_stats = {}
    compare_inst_stats = None

    with open(llvm_file, 'r', encoding='utf-8') as file:
        # parse preamble
        for line in file:
            line = line.lstrip()
            if line.startswith(';'):
                continue

            if line.startswith('source_filename'):
                continue

            target_prefix = 'target triple = "'
            if line.startswith(target_prefix):
                for target in inst_stats.keys():
                    if line[len(target_prefix):].startswith(target):
                        current_isa = target
                        # print('Found target isa is', current_isa)
                if current_isa is not None:
                    break

            if line.startswith('target'):
                continue

        # no matching inst_stats found
        if current_isa is None:
            sys.exit(0)

        compare_inst_stats = inst_stats[current_isa]
        for key in compare_inst_stats.keys():
            current_stats[key] = 0

        # process remaining file
        for line in file:
            line = line.lstrip()
            if line.startswith('declare'):
                continue

            if line.startswith('define'):
                continue

            if line.startswith(';'):
                continue

            if line.startswith('!'):
                continue

            for key in compare_inst_stats.keys():
                if key not in line:
                    continue

                current_stats[key] += 1

    # print("current_stats:", current_stats)

    passed = True

    for key in compare_inst_stats.keys():
        passed = current_stats[key] == compare_inst_stats[key]
        if not passed:
            break

    if passed:
        print('passed!')
    else:
        print('failed with wrong instructions!')

    for key in compare_inst_stats.keys():
        if current_stats[key] != compare_inst_stats[key]:
            print('(!)', end=' ')
        else:
            print('   ', end=' ')
        print(key, '(', compare_inst_stats[key], ') ->', current_stats[key])

    if not passed:
        sys.exit(-1)
