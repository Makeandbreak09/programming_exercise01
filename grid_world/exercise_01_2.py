# import grid world
from gridworld import GridWorld

# initialize grid world environment with given size, battery position and obstacle positions
N = 6
battery_pos_ = (4,3)
obstacles_ = {(0, 0), (1, 1), (1, 2), (1, 4), (2, 5), (3, 3), (4, 0), (4, 2), (4, 4)}
env = GridWorld(size=N, battery_pos=battery_pos_, obstacles=obstacles_)
        
# initialize state
state = (5,3)

# test render function
env.render(state)

print("States: ", env.states())
print("Successors: ", env.successors(state))

print("\nTransitions and Rewards:")
for a_idx, action_name in enumerate(env.actions):
    for succ in env.successors(state):
        prob = env.p(succ, state, a_idx)
        if prob > 0:
            rew = env.reward(succ, state, a_idx)
            print(f"Action: {action_name:5} | Successor: {succ} | Prob: {prob} | Reward: {rew}")