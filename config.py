# Hyperparameters for DQN agent, memory and training
EPISODES = 3500
HEIGHT = 84
WIDTH = 84
HISTORY_SIZE = 4
learning_rate = 0.0001
evaluation_reward_length = 100
Memory_capacity = 1000000
train_frame = 100000 # You can set it to a lower value while testing your code so you don't have to wait longer to see if the training code does not have any syntax errors
batch_size = 128
scheduler_gamma = 0.4
scheduler_step_size = 100000

# Hyperparameters for Double DQN agent
update_target_network_frequency = 1000

# Hyperparameters for DQN LSTM agent
lstm_seq_length = 5