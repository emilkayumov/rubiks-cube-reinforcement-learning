import argparse
import pickle
import time

import numpy as np
import pycuber
import tqdm

import cube
import dqn
import solver

POLICY_PATH = \
    'weights/dqn_mean_100_gamma_0.9_batch_512_episodes_100000.pkl'
SEED = 45


def get_policy():
    policy = dqn.DQN(n_space=cube.N_SPACE, n_action=cube.N_ACTION)
    policy.load_state_dict(pickle.load(open(POLICY_PATH, 'rb')))
    return policy


def get_value():
    return dqn.SimpleValue()


def revert_actions(actions):
    def add_revert(action):
        if action.endswith("'"):
            return action[:-1]
        else:
            return action + "'"

    actions = [add_revert(x) for x in actions[::-1]]
    return actions


def test_solver(steps, solver, n_iter, time_limit):
    np.random.seed(SEED)

    result = {
        'success': [],
        'actions': [],
        'time': [],
        'depths': [],
        'values': [],
        'real actions': []}

    for _ in tqdm.tqdm(range(n_iter)):
        solving_cube = pycuber.Cube()
        random_actions = np.random.choice(cube.ACTIONS, size=steps)
        solving_cube.perform_algo(random_actions)

        start_time = time.time()
        is_done, actions, depth, value = \
            solver.solve(solving_cube, time_limit=time_limit)
        solve_time = time.time() - start_time

        result['success'].append(is_done)
        result['actions'].append(actions)
        result['time'].append(solve_time)
        result['depths'].append(depth)
        result['values'].append(value)
        result['real actions'].append(revert_actions(random_actions))

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', '-i', type=int)
    parser.add_argument('--steps', '-s', type=int)
    parser.add_argument('--method', '-m', type=str)
    parser.add_argument('--time-limit', '--tl', type=int)
    parser.add_argument('--temperature', '-t', type=float)
    parser.add_argument('--c-puct', '-c', type=float)
    parser.add_argument('--result', type=str)
    args = parser.parse_args()

    if args.method == 'greedy':
        solver_ = solver.GreedySolver(get_policy())
    elif args.method == 'naive':
        solver_ = solver.SimpleMCTSSolver(
            get_policy(), temperature=args.temperature)
    elif args.method == 'ucb':
        solver_ = solver.UCBSolver(
            get_policy(), get_value(), c_puct=args.c_puct, temperature=args.temperature)
    else:
        raise ValueError('unknown solver')

    result = test_solver(
        steps=args.steps, solver=solver_, n_iter=args.iterations,
        time_limit=args.time_limit)

    result['parameters'] = {
        'iterations': args.iterations,
        'steps': args.steps,
        'method': args.method,
        'time limit': args.time_limit,
        'policy tau': args.policy_tau,
        'ucb c': args.ucb_c}

    print('success rate: {}%'.format(100 * np.mean(result['success'])))
    pickle.dump(result, open(args.result, 'wb'))
