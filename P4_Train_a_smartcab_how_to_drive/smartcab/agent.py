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
        self.alpha = 0.2
        # self.gamma = 0

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

        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], inputs['right'], self.next_waypoint)
        
        # TODO: Select action according to your policy
        # moves = [None, 'forward', 'left', 'right']
        # action = random.choice(moves)

        possible_actions = {action: self.Q_learner.get((self.state, action), 0) for action in self.valid_actions}

        # action = max(possible_actions.iteritems(), key=lambda x:x[1])[0]

        actions = [action for action in self.valid_actions if possible_actions[action] == max(possible_actions.values())]
        action = random.choice(actions)

        # Execute action and get reward
        reward = self.env.act(self, action)
        self.reward += reward

        # possibl e_actions_prime = {action: self.Q_learner.get((self.state, action), 0) for action in self.valid_actions}

        # actions_prime = [action for action in self.valid_actions if possible_actions_prime[action] == max(possible_actions_prime.values())]
        # action_prime = random.choice(actions_prime)

        # print possible_actions
        # print action
        # print possible_actions_prime
        # print action_prime

        # TODO: Learn policy based on state, action, reward
        # self.state_prime = (('light', inputs['light']), ('next_waypoint', self.next_waypoint))
        
        # self.Q_learner[(self.state, action)] = \
        #             (1-self.alpha) * \
        #             self.Q_learner.get((self.state, action), 0) + \
        #             self.alpha * \
        #             (reward + \
        #                 self.gamma * self.Q_learner.get((self.state_prime, action_prime), 0)
        #                 )
        # print self.Q_learner[(self.state, action)]

        self.Q_learner[(self.state, action)] = \
                    (1-self.alpha) * \
                    self.Q_learner.get((self.state, action), 0) + \
                    self.alpha * reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100000)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
