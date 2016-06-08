import random
import numpy as np
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Q_learner = {}
        self.reward = 0
        self.alpha = 0.8
        self.gamma = 0.2


    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.reward = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update state
        self.valid_actions = self.env.valid_actions

        self.state = (('light', inputs['light']), ('next_waypoint', self.next_waypoint))

        if self.state not in self.Q_learner:
            possible_actions = {possible_action: 0 for possible_action in self.valid_actions}
            self.Q_learner[self.state] = possible_actions

        # TODO: Select action according to your policy
        # moves = [None, 'forward', 'left', 'right']
        # action = random.choice(moves)
        if np.random.random() > 0.05:
            for value in self.Q_learner[self.state].values():
                if value != 0:
                    action = max(self.Q_learner[self.state].iteritems(), key=lambda x:x[1])[0]
            action = random.choice(self.valid_actions)
        else:
            action = random.choice(self.valid_actions)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.reward += reward

        # TODO: Learn policy based on state, action, reward
        if self.state not in self.Q_learner:
            possible_actions = {possible_action: 0 for possible_action in self.valid_actions}
            self.Q_learner[self.state] = possible_actions

        self.Q_learner[self.state][action] = (1-self.alpha) * \
                    self.Q_learner[self.state][action] + \
                    self.alpha * \
                    (reward + 
                        self.gamma * max(self.Q_learner[self.state].values())
                        )

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e)
    sim.run(n_trials=100)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
