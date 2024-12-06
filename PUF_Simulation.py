import numpy as np
import matplotlib.pyplot as plt

# Generate challenge-response pairs (CRPs)
num_crps = 10000
challenge = np.random.randint(0, 2, (num_crps, 64))  # 64-bit challenges
response = np.random.randint(0, 2, (num_crps, 64))   # 64-bit responses

# Add noise to simulate real-world conditions
noise = np.random.normal(0, 0.1, response.shape)  # Adjusted noise scale for noticeable effect
response_noisy = response + noise
response_noisy = np.clip(response_noisy, 0, 1)    # Ensure values stay between 0 and 1
response_noisy = np.where(response_noisy >= 0.5, 1, 0)  # Binary thresholding

# Evaluate classification accuracy
correct = np.sum(np.all(response == response_noisy, axis=1))  # Count correctly matched responses
accuracy = (correct / num_crps) * 100
print(f"PUF Simulation Accuracy: {accuracy:.2f}%")

# Analyze noise impact
bit_flips = np.sum(response != response_noisy, axis=1)  # Number of flipped bits per response
avg_bit_flips = np.mean(bit_flips)  # Average number of bit flips per challenge-response pair
print(f"Average Bit Flips per Response: {avg_bit_flips:.2f}")

# Visualization 1: Distribution of Bit Flips
plt.figure(figsize=(8, 6))
plt.hist(bit_flips, bins=range(0, 65), color='skyblue', edgecolor='black', alpha=0.7)
plt.title("Distribution of Bit Flips")
plt.xlabel("Number of Bit Flips")
plt.ylabel("Frequency")
plt.grid(axis='y')
plt.show()

# Visualization 2: Original vs. Noisy Responses
# Select a random subset for visualization
subset_indices = np.random.choice(num_crps, 10, replace=False)
subset_responses = response[subset_indices]
subset_noisy_responses = response_noisy[subset_indices]

plt.figure(figsize=(10, 6))
for i, (orig, noisy) in enumerate(zip(subset_responses, subset_noisy_responses)):
    plt.subplot(5, 2, i + 1)
    plt.bar(range(64), orig, alpha=0.7, label='Original', color='blue')
    plt.bar(range(64), noisy, alpha=0.5, label='Noisy', color='red')
    plt.title(f"CRP {i+1}")
    plt.xticks([])
    plt.yticks([0, 1])
    if i == 0: plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# Visualization 3: Accuracy Over Noise Levels
noise_levels = np.linspace(0, 0.5, 20)
accuracies = []
for noise_std in noise_levels:
    noisy = response + np.random.normal(0, noise_std, response.shape)
    noisy = np.clip(noisy, 0, 1)
    noisy = np.where(noisy >= 0.5, 1, 0)
    acc = np.sum(np.all(response == noisy, axis=1)) / num_crps * 100
    accuracies.append(acc)

plt.figure(figsize=(8, 6))
plt.plot(noise_levels, accuracies, marker='o', color='green')
plt.title("PUF Accuracy vs. Noise Level")
plt.xlabel("Noise Standard Deviation")
plt.ylabel("Accuracy (%)")
plt.grid()
plt.show()
