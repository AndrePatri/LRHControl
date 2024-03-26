import torch
from ppo_tanhv2 import ActorCriticThB

obs_dim = 3  # Example observation dimension
actions_dim = 10 # Example action dimension
n_tanh_outputs = 2  # Example number of tanh outputs
model = ActorCriticThB(obs_dim, actions_dim, n_tanh_outputs, tanh_lb = 0.0, tanh_ub = 0.5)

# Create a dummy input tensor
dummy_input = torch.tensor([[0.1, -0.1, 23]])  # Assuming batch size of 1

# Call the get_action method
action, log_prob, entropy = model.get_action(dummy_input)

# Print the results
print("Action:")
print(action)
print("Log Probability:")
print(log_prob)
print("Entropy:")
print(entropy)
