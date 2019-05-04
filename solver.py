import time

import numpy as np

import cube

ITER_LIMIT = 1000000000
EPS_N = 0.01
DEBUG = 0


class GreedySolver:
    def __init__(self, policy):
        self.policy = policy

    def solve(self, solving_cube, time_limit=None, iter_limit=None):
        if time_limit is None and iter_limit is None:
            raise ValueError('infinite loop maybe')

        if iter_limit is None:
            iter_limit = ITER_LIMIT

        solving_cube = solving_cube.copy()
        start_time = time.time()
        actions = []

        for i_current in range(iter_limit):
            state = cube.get_state(solving_cube)
            observation = cube.get_observation(state)
            if cube.is_done(state):
                return True, actions, i_current, None

            action = cube.ACTIONS[self.policy.select_action(observation)]
            actions.append(action)
            solving_cube.perform_step(action)

            if time_limit and time.time() - start_time > time_limit:
                return False, None, i_current, None

        else:
            return False, None, iter_limit, None


class SimpleMCTSSolver:
    def __init__(self, policy, temperature):
        self.root = None
        self.policy = policy
        self.temperature = temperature

    def solve(self, solving_cube, time_limit=None, iter_limit=None):
        if time_limit is None and iter_limit is None:
            raise ValueError('infinite loop maybe')

        self.root = SimpleNode(solving_cube, self, 0)

        if iter_limit is None:
            iter_limit = ITER_LIMIT

        start_time = time.time()

        for i_current in range(iter_limit):
            iter_time = time.time()
            is_done, actions, depth = self.root.update()
            if is_done:
                return True, actions, depth, None

            if time_limit and time.time() - start_time > time_limit:
                return False, None, depth, None

        else:
            return False, None, None, None


class SimpleNode:
    def __init__(self, node_cube, mcts, depth):
        self.cube = node_cube
        self.mcts = mcts
        self.depth = depth

        self.is_leaf = True
        self.is_done = cube.is_done(cube.get_state(node_cube))
        self.probabilities = None
        self.children = [None] * cube.N_ACTION

    def update(self):
        if self.is_leaf:
            if self.is_done:
                return True, [], self.depth
            self.probabilities = self.mcts.policy.get_action_probabilities(
                cube.get_observation(cube.get_state(self.cube)),
                self.mcts.temperature)
            self.is_leaf = False
            return False, None, self.depth

        else:
            action = self.probabilities.sample()
            if self.children[action] is None:
                self.children[action] = SimpleNode(
                    self.cube.copy().perform_step(cube.ACTIONS[action]),
                    self.mcts, self.depth + 1)
            is_done, actions, max_depth = self.children[action].update()

            if is_done:
                return True, [cube.ACTIONS[action]] + actions, max_depth
            else:
                return False, None, max_depth


class UCBSolver:
    def __init__(self, policy, value, c_puct, temperature):
        self.policy_f = policy
        self.value_f = value
        self.temperature = temperature
        self.c_puct = c_puct

    def solve(self, solving_cube, time_limit=None, iter_limit=None):
        if time_limit is None and iter_limit is None:
            raise ValueError('infinite loop maybe')

        root = UBCNode(solving_cube, self, depth=0, c_puct=self.c_puct)

        if iter_limit is None:
            iter_limit = ITER_LIMIT

        max_depth = 0
        start_time = time.time()

        for i_current in range(iter_limit):
            iter_time = time.time()
            is_done, actions, depth, value = root.update()
            max_depth = max(max_depth, depth)
            if is_done:
                return True, actions, max_depth, value

            if time_limit and time.time() - start_time > time_limit:
                return False, None, max_depth, value
            if DEBUG:
                print('iter: {} \ttime: {}\n'.format(
                    i_current, time.time() - iter_time))

        else:
            return False, None, None, None


class UBCNode:
    def __init__(self, node_cube, mcts, depth, c_puct):
        self.cube = node_cube
        self.mcts = mcts
        self.depth = depth
        self.c_puct = c_puct

        state = cube.get_state(node_cube)
        self.is_done = cube.is_done(state)
        self.value = self.mcts.value_f.get_value(state)

        self.visit_count = 0
        self.is_leaf = True
        self.probabilities = None
        self.children = [None] * cube.N_ACTION

    def get_max_ubc_action(self):
        if DEBUG:
            print('calculate max ubc action')

        if DEBUG:
            for i_child in range(len(self.children)):
                print('action={}\tv+u={}\tv={}\tvisit count={}\tu={}'.format(
                    cube.ACTIONS[i_child],
                    round(float(
                        self.children[i_child].value +
                        self.c_puct * self.probabilities.probs[i_child] *
                        np.sqrt(np.log(self.visit_count) /
                                (self.children[i_child].visit_count + EPS_N))),
                        3),
                    round(self.children[i_child].value, 3),
                    self.children[i_child].visit_count,
                    round(float(
                        self.c_puct * self.probabilities.probs[i_child] *
                        np.sqrt(np.log(self.visit_count) /
                                (self.children[i_child].visit_count + EPS_N))),
                        3)))

        return np.argmax([
            self.children[i_child].value +
            self.c_puct * self.probabilities.probs[i_child] *
            np.sqrt(np.log(self.visit_count) /
                    (self.children[i_child].visit_count + EPS_N))
            for i_child in range(len(self.children))])

    def update(self):
        self.visit_count = self.visit_count + 1
        if DEBUG:
            print('updating node, depth={}\tvisit count={}'.format(
                self.depth, self.visit_count))

        if self.is_leaf:
            if self.is_done:
                return True, [], self.depth, self.value

            self.probabilities = self.mcts.policy_f.get_action_probabilities(
                cube.get_observation(cube.get_state(self.cube)),
                self.mcts.temperature)
            if DEBUG:
                print('node is leaf, init all children')
                print('probs: {}'.format(' '.join(list(map(
                    lambda x: str(round(x, 2)),
                    self.probabilities.probs.numpy())))))

            self.is_leaf = False
            self.children = [
                UBCNode(
                    self.cube.copy().perform_step(action),
                    self.mcts, self.depth + 1, self.c_puct)
                for action in cube.ACTIONS]

            return False, None, self.depth, self.value

        else:
            action = self.get_max_ubc_action()
            if DEBUG:
                print('next action {}'.format(cube.ACTIONS[action]))
            is_done, actions, max_depth, value = \
                self.children[action].update()
            if DEBUG:
                print('explored action {}\tv={}\tdepth={}\tmax_depth={}'.format(
                    cube.ACTIONS[action], value, self.depth + 1, max_depth))

            if is_done:
                return (
                    True, [cube.ACTIONS[action]] + actions, max_depth,
                    max(self.value, value))
            else:
                return False, None, max_depth, max(self.value, value)
