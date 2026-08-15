# import os
# import sys
# import argparse
#
# prj_path = os.path.join(os.path.dirname(__file__), '..')
# if prj_path not in sys.path:
#     sys.path.append(prj_path)
#
# from lib.test.evaluation import get_dataset
# from lib.test.evaluation.running import run_dataset
# from lib.test.evaluation.tracker import Tracker
#
#
# def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
#                 num_gpus=8):
#     """Run tracker on sequence or dataset.
#     args:
#         tracker_name: Name of tracking method.
#         tracker_param: Name of parameter file.
#         run_id: The run id.
#         dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
#         sequence: Sequence number or name.
#         debug: Debug level.
#         threads: Number of threads.
#     """
#
#     dataset = get_dataset(dataset_name)
#
#     if sequence is not None:
#         dataset = [dataset[sequence]]
#
#     trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]
#     run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus
#     )
#
#
# def main():
#     parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
#     parser.add_argument('tracker_name', type=str, help='Name of tracking method.')
#     parser.add_argument('tracker_param', type=str, help='Name of config file.')
#     parser.add_argument('--runid', type=str, default=None, help='The run id.')
#     parser.add_argument('--dataset_name', type=str, default='otb', help='Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).')
#     parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
#     parser.add_argument('--debug', type=int, default=0, help='Debug level.')
#     parser.add_argument('--threads', type=int, default=6, help='Number of threads.')
#     parser.add_argument('--num_gpus', type=int, default=1)
#
#     args = parser.parse_args()
#
#     try:
#         seq_name = int(args.sequence)
#     except:
#         seq_name = args.sequence
#
#     run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
#                 args.threads, num_gpus=args.num_gpus)
#
#
# if __name__ == '__main__':
#     main()
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import argparse
import time
import torch
from concurrent.futures import ThreadPoolExecutor
from lib.test.evaluation import get_dataset
from lib.test.evaluation.tracker import Tracker
from lib.test.evaluation.running import run_dataset


def run_with_shared_tracker(seq_name, tracker, dataset_name, debug, threads):
    print(f"[INFO] Starting sequence: {seq_name}")
    start_time = time.time()

    dataset = get_dataset(dataset_name)
    seqs = [s for s in dataset if s.name == seq_name]
    if not seqs:
        print(f"[WARN] Sequence {seq_name} not found.")
        return

    run_dataset(seqs, [tracker], debug=debug, threads=threads, num_gpus=1)

    elapsed = time.time() - start_time
    print(f"[DONE] Sequence {seq_name} finished in {elapsed:.2f} seconds.")

    # Log timing info to file (optional)
    os.makedirs('timing_logs', exist_ok=True)
    with open(f'timing_logs/{seq_name}.log', 'w') as f:
        f.write(f"{seq_name}: {elapsed:.2f} seconds\n")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker_name', type=str, required=True)
    parser.add_argument('--tracker_param', type=str, required=True)
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--run_id', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0, help='Number of parallel threads. If 0, run serially.')
    parser.add_argument('--threads', type=int, default=2, help='Threads per process for loading data')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = get_dataset(args.dataset_name)
    sequence_names = [seq.name for seq in dataset]

    total_start = time.time()

    tracker = Tracker(args.tracker_name, args.tracker_param, args.dataset_name, args.run_id)

    if args.num_workers <= 1:
        print("[INFO] Running in serial mode...")
        run_dataset(dataset, [tracker], debug=args.debug, threads=args.threads, num_gpus=1)
    else:
        print(f"[INFO] Running in threaded parallel mode with {args.num_workers} threads (1 seq per thread)...")

        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [
                executor.submit(run_with_shared_tracker, seq_name, tracker, args.dataset_name, args.debug, args.threads)
                for seq_name in sequence_names
            ]

            for future in futures:
                future.result()

    total_elapsed = time.time() - total_start
    print(f"[FINISHED] Total testing time: {total_elapsed:.2f} seconds.")
