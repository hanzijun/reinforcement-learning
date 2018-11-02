from maze_env import Maze
from RL_brain import DeepQNetwork
from RNNBrain import RNNNetwork
import numpy as np
from DDPGBrain import DDPG

def run_maze():
    step = 0
    for episode in range(200):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()
            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if (step > 200) and (step % 5 == 0):
                RL.learn()
            # swap observation
            observation = observation_
            # break while loop when end of this episode
            if done:
                break
            step += 1
    # end of game
    print('game over')
    RL.saver.save(RL.sess, "./eval_network/rlnew.ckpt")
    env.destroy()
if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DDPG(env.n_actions, env.n_features,
                      memory_size=2000,
                      output_graph=True,
                      param_file= None
                      )
   # print env.n_actions
    #print env.n_features
    env.after(100, run_maze)
    env.mainloop()
