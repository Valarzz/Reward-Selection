import os
import pickle
import numpy as np
import torch
import random
import time
import subprocess
from argparse import ArgumentParser
from minatar import Environment
from iql import ImplicitQLearning, TwinQ, ValueFunction
from bc import BehaviorCloningModel
from minatar.heuristics import heuristicselection, DEVICE


def control_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_budget_list(game):
    if game == 'breakout':
        return [2000, 4000, 6000, 8000, 10000, 12000, 13731]
    elif game == 'freeway':
        return [120000, 240000, 360000, 480000, 600000, 720000]
    elif game == 'seaquest':
        return [96000, 192000, 288000, 384000, 480000, 576000]
    elif game == 'asterix':
        return [72000, 144000, 216000, 288000, 360000, 432000]
    else:
        raise ValueError(f"Unknown game: {game}")

# def save_file_es(args, budget, Js, dist_idx, iter_idx):
#     savepath = f"{args.root}/{args.game}/result/{args.expname}/es/{budget}/"
#     os.makedirs(savepath, exist_ok=True)
#     np.save(savepath+f"dist{dist_idx}_iter{iter_idx}.npy", np.array(Js))

def submit_job(job_args, name):
    sbatch_cmd = [
        "sbatch",
        "--partition=cpu",
        "--cpus-per-task=4",
        "--mem=32G",
        "--time=03:00:00",
        f"--output=output/log_{name}.out",
        "--wrap",
        f"python minatar/esjob.py {job_args}"
    ]
    output = subprocess.check_output(sbatch_cmd).decode()
    jobid = int(output.strip().split()[-1])
    return jobid

def is_job_running(jobid):
    try:
        output = subprocess.check_output(["squeue", "-j", str(jobid)])
        return len(output.decode().strip().splitlines()) > 1
    except Exception:
        return False

def main(args):
    control_seed(args.seed)
    st = time.time()
    print("Loading dataset ...")
    loadpath = f"{args.root}/{args.game}/dataset/{args.game}_0.5"
    with open(loadpath, "rb") as f:
        dataset = pickle.load(f)
    print(f"Done: takes {time.time() - st}s")

    env = Environment(args.game)
    in_channels = env.state_shape()[2]
    num_actions = env.num_actions()

    iql = ImplicitQLearning(
        qf=TwinQ(in_channels, num_actions),
        vf=ValueFunction(in_channels),
        optimizer_factory=lambda params: torch.optim.Adam(params, lr=args.learning_rate),
        num_actions=num_actions,
        tau=args.tau,
        beta=args.beta,
        alpha=args.alpha,
        discount=args.discount,
    ).to(DEVICE)
    bc = BehaviorCloningModel(in_channels, num_actions).to(DEVICE)

    selectionagent = heuristicselection(iql=iql,
                 bc=bc,
                 dataset=dataset, 
                 env=env, 
                 budget=args.budget, 
                 each_query=args.each_query,
                 sdim=in_channels,
                 adim=num_actions,
                 N=1000, 
                 gamma=args.discount, 
                 blind=False,  # ES is not blind
                 uniform=False, 
                 decay='linear',
                 zero='vis',
                 fixeddecay=False,
                 fixtime=0.5,
                 train_steps=args.train_steps,  # 200_000
                 eval_period=args.eval_period,
                 batch_size=args.batch_size,
                 n_eval_episodes=args.n_eval_episodes,
                 decay_temp=args.decay_temp,
                 initial_sample=args.initial_sample,
                 game=args.game,
                 )

    total_states = selectionagent.total_states
    state_indices = np.arange(total_states)
    budget_list = get_budget_list(args.game)
    if args.budget not in budget_list:
        raise ValueError(f"Budget {args.budget} not in allowed list for {args.game}: {budget_list}")
    budget = args.budget

    # ES parameters
    M = args.M  # number of distributions per iteration
    K = args.K  # number of ES iterations
    E = args.E  # number of elites
    b = budget  # number of states to select per distribution
    MAX_TRIES = 1  # max number of tries for each job               

    # Initialize mean and std
    mean_path = f"{args.root}/{args.game}/result/{args.expname}/es_checkpoint/{budget}_mean.npy"
    std_path = f"{args.root}/{args.game}/result/{args.expname}/es_checkpoint/{budget}_std.npy"
    os.makedirs(os.path.dirname(mean_path), exist_ok=True)
    if os.path.exists(mean_path) and os.path.exists(std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)
        print(f"[Resume] Loaded mean and std from checkpoint {mean_path} and {std_path}.")
    else:
        mean = np.ones(total_states) * (b / total_states)
        std = np.ones(total_states) * 0.1

    for iter_idx in range(K):
        print(f"ES Iteration {iter_idx+1}/{K}")
        all_selected = []
        all_args = []
        all_names = []
        job_status = {}  # m: {"jobid": ..., "tries": ...}
        result_dir = f"{args.root}/{args.game}/result/{args.expname}/es/{budget}/"
        for m in range(M):
            result_path = os.path.join(result_dir, f"dist{m}_iter{iter_idx}.npy")
            if os.path.exists(result_path):
                print(f"[Resume] Result for dist{m}_iter{iter_idx} already exists, skipping job launch.")
                all_selected.append(None)
                all_args.append(None)
                all_names.append(None)
                continue
            dist = np.random.normal(mean, std)
            top_indices = np.argsort(dist)[-b:][::-1]
            all_selected.append(top_indices)
            indices_path = f"{args.root}/{args.game}/result/{args.expname}/es_indices/{budget}_iter{iter_idx}_dist{m}.npy"
            os.makedirs(os.path.dirname(indices_path), exist_ok=True)
            np.save(indices_path, top_indices)
            job_args = f"--game {args.game} --budget {budget} --expname {args.expname} --es_indices {indices_path} --dist_idx {m} --iter_idx {iter_idx} --learning_rate {args.learning_rate} --alpha {args.alpha} --tau {args.tau} --beta {args.beta} --discount {args.discount} --train_steps {args.train_steps} --batch_size {args.batch_size} --eval_period {args.eval_period} --n_eval_episodes {args.n_eval_episodes} --seed {args.seed}"
            all_args.append(job_args)
            name = f"es_{args.game}_iter{iter_idx}_dist{m}"
            all_names.append(name)
            jobid = submit_job(job_args, name)
            job_status[m] = {"jobid": jobid, "tries": 1}

        while True:
            finished = [os.path.exists(os.path.join(result_dir, f"dist{m}_iter{iter_idx}.npy")) if all_selected[m] is not None else True for m in range(M)]
            if all(finished):
                break
            missing = [m for m in range(M) if not finished[m] and all_selected[m] is not None]
            print(f"Waiting for jobs to finish: {sum(finished)}/{M} done. Missing: {missing}")
            for m in missing:
                result_path = os.path.join(result_dir, f"dist{m}_iter{iter_idx}.npy")
                if os.path.exists(result_path):
                    continue
                jobid = job_status[m]["jobid"]
                tries = job_status[m]["tries"]
                if not is_job_running(jobid):
                    if tries < MAX_TRIES:
                        print(f"Job {jobid} for m={m} finished but no result file. Resubmitting (try {tries+1}/{MAX_TRIES})...")
                        jobid = submit_job(all_args[m], all_names[m])
                        job_status[m]["jobid"] = jobid
                        job_status[m]["tries"] += 1
                    else:
                        print(f"Job for m={m} failed after {MAX_TRIES} tries. Skipping this job for this iteration.")
                        finished[m] = True
                        all_selected[m] = None  # Mark as skipped
            if all(finished):
                break
            time.sleep(60)

        # Read all results
        all_returns = []
        for m in range(M):
            if all_selected[m] is None:
                all_returns.append(float('-inf'))  # Skipped jobs get -inf so they are not selected as elite
                continue
            result_path = os.path.join(result_dir, f"dist{m}_iter{iter_idx}.npy")
            Js = np.load(result_path)
            all_returns.append(np.mean(Js))
        all_returns = np.array(all_returns)
        elite_indices = np.argsort(all_returns)[-E:][::-1]
        elite_selected = [all_selected[i] for i in elite_indices if all_selected[i] is not None]
        if len(elite_selected) == 0:
            print("No elites found for this iteration. Skipping update.")
            continue
        elite_matrix = np.stack([np.isin(state_indices, sel).astype(float) for sel in elite_selected])
        mean = elite_matrix.mean(axis=0)
        std = elite_matrix.std(axis=0) + 1e-6
        print(f"Elite mean: {mean.mean():.4f}, std: {std.mean():.4f}")

        # Save mean and std after each iteration
        np.save(mean_path, mean)
        np.save(std_path, std)
        # Also save mean for each k for future use
        mean_k_path = f"{args.root}/{args.game}/result/{args.expname}/es_checkpoint/{budget}_mean_{iter_idx}.npy"
        np.save(mean_k_path, mean)

    print("ES finished.")

    # Final evaluation: sample 10 state sets from mean, push 10 jobs, then collect results
    final_eval_dir = f"{args.root}/{args.game}/result/{args.expname}/es_final_eval/{budget}/"
    os.makedirs(final_eval_dir, exist_ok=True)

    # Submit one job for the top b indices from the mean vector (not a sampled distribution)
    mean_topb_indices = np.argsort(mean)[-b:][::-1]
    mean_topb_path = os.path.join(final_eval_dir, "mean_topb.npy")
    np.save(mean_topb_path, mean_topb_indices)
    mean_job_args = f"--game {args.game} --budget {budget} --expname {args.expname} --es_indices {mean_topb_path} --dist_idx 999 --iter_idx 9999 --learning_rate {args.learning_rate} --alpha {args.alpha} --tau {args.tau} --beta {args.beta} --discount {args.discount} --train_steps {args.train_steps} --batch_size {args.batch_size} --eval_period {args.eval_period} --n_eval_episodes {args.n_eval_episodes} --seed {args.seed}"
    mean_job_name = f"esfinal_{args.game}_budget{budget}_mean_topb"
    # print(mean_job_args, mean_job_name)
    # breakpoint()
    mean_jobid = submit_job(mean_job_args, mean_job_name)
    print(f"Submitted mean_topb job with jobid {mean_jobid}")

    final_state_sets = []
    final_names = []
    final_job_status = {}
    final_job_args = {}
    final_job_names = {}
    for i in range(10):
        result_path = os.path.join(final_eval_dir, f"dist{i}_iter_final.npy")
        if os.path.exists(result_path):
            print(f"[Resume] Final eval result for set {i} already exists, skipping job launch.")
            final_state_sets.append(None)
            final_names.append(None)
            continue
        dist = np.random.normal(mean, std)
        top_indices = np.argsort(dist)[-b:][::-1]
        final_state_sets.append(top_indices)
        indices_path = os.path.join(final_eval_dir, f"final_stateset_{i}.npy")
        np.save(indices_path, top_indices)
        job_args = f"--game {args.game} --budget {budget} --expname {args.expname} --es_indices {indices_path} --dist_idx {i} --iter_idx 9999 --learning_rate {args.learning_rate} --alpha {args.alpha} --tau {args.tau} --beta {args.beta} --discount {args.discount} --train_steps {args.train_steps} --batch_size {args.batch_size} --eval_period {args.eval_period} --n_eval_episodes {args.n_eval_episodes} --seed {args.seed}"
        name = f"esfinal_{args.game}_budget{budget}_set{i}"
        final_names.append(name)
        final_job_args[i] = job_args
        final_job_names[i] = name
        jobid = submit_job(job_args, name)
        final_job_status[i] = {"jobid": jobid, "tries": 1}

    while True:
        finished = [os.path.exists(os.path.join(final_eval_dir, f"dist{i}_iter_final.npy")) if final_state_sets[i] is not None else True for i in range(10)]
        if all(finished):
            break
        missing = [i for i in range(10) if not finished[i] and final_state_sets[i] is not None]
        print(f"Waiting for final eval jobs: {sum(finished)}/10 done. Missing: {missing}")
        for i in missing:
            result_path = os.path.join(final_eval_dir, f"dist{i}_iter_final.npy")
            if os.path.exists(result_path):
                continue
            jobid = final_job_status[i]["jobid"]
            tries = final_job_status[i]["tries"]
            if not is_job_running(jobid):
                if tries < MAX_TRIES:
                    print(f"Final eval job {jobid} for set={i} finished but no result file. Resubmitting (try {tries+1}/{MAX_TRIES})...")
                    jobid = submit_job(final_job_args[i], final_job_names[i])
                    final_job_status[i]["jobid"] = jobid
                    final_job_status[i]["tries"] += 1
                else:
                    print(f"Final eval job for set={i} failed after {MAX_TRIES} tries. Skipping.")
                    finished[i] = True
                    final_state_sets[i] = None
        if all(finished):
            break
        time.sleep(30)

    final_means = []
    for i in range(10):
        if final_state_sets[i] is None:
            final_means.append(float('-inf'))
            continue
        result_path = os.path.join(final_eval_dir, f"dist{i}_iter_final.npy")
        Js = np.load(result_path)
        final_means.append(np.mean(Js))
    np.save(os.path.join(final_eval_dir, "all_final_means.npy"), np.array(final_means))
    print(f"Final evaluation means: {final_means}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--expname', type=str, default="esdebug")
    parser.add_argument('--game', type=str, default="breakout")
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--alpha', type=float, default=0.005)
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=3.0)
    parser.add_argument('--discount', type=float, default=0.99)
    parser.add_argument('--train_steps', type=int, default=125_000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_period', type=int, default=500)
    parser.add_argument('--n_eval_episodes', type=int, default=50)
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--each_query', type=int, default=250)
    parser.add_argument('--decay_temp', type=float, default=6.)
    parser.add_argument('--initial_sample', type=int, default=250)
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--M', type=int, default=100)
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--E', type=int, default=10)
    args = parser.parse_args()
    print(args)
    print(DEVICE)
    main(args) 