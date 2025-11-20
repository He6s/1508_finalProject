import os
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

#mean returns

methods = ["Random baseline", "Bandit baseline", "RL λ=0", "RL λ=0.2"]

mean_returns = [
    21.400,   # random_baseline_long
    21.290,   # bandit_baseline_long
    137.450,  # lambda_0_long
    141.600,  # lambda_0_2_long
]

plt.figure()
plt.bar(methods, mean_returns)
plt.ylabel("Mean return")
plt.title("Mean return by method")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "mean_return_bar.png"))

# diversity for baselines

baseline_methods = ["Random baseline", "Bandit baseline"]

mean_diversities = [
    0.080,  # random_baseline_long
    0.070,  # bandit_baseline_long
]

plt.figure()
plt.bar(baseline_methods, mean_diversities)
plt.ylabel("Mean diversity")
plt.title("Slate diversity (baselines)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "diversity_bar.png"))

print("Saved plots to:")
print(f"  {os.path.join(RESULTS_DIR, 'mean_return_bar.png')}")
print(f"  {os.path.join(RESULTS_DIR, 'diversity_bar.png')}")
