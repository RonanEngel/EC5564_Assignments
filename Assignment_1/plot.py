import numpy as np
import matplotlib.pyplot as plt
import random
import scipy
from scipy.stats import multivariate_normal


# Sorry for the typos in advance

print("Installations Working!")
# Parameter Definitions

class_1_prior = .3
class_2_prior = .3
class_3_prior = .4

# Set Random Seed
rng = np.random.default_rng(1)

dist_1_mu = [1, 1, 1]
dist_2_mu = [1, -1, -1]
dist_3_mu = [-1, 1, -1]
dist_4_mu = [-1, -1, 1]


# Quick verification of mean equidistance
mu_list = [dist_1_mu, dist_2_mu, dist_3_mu, dist_4_mu]


dist_1_cov = [[2,   -1.5, .5],
              [-1.5, 2, 1],
              [.5, 1, 5]]

dist_2_cov = [[1,   -.2, -.5],
              [-.2, 2, .5],
              [-.5, .5, 1.5]]

dist_3_cov = [[1, .1, .5],
              [.1, 1.5, 1],
              [.5, 1, 1.5]]

dist_4_cov = [[2, .1, .5],
              [.1, 2, .1],
              [.5, .1, .5]]

cov_list = [dist_1_cov, dist_2_cov, dist_3_cov, dist_4_cov]

# average standard deviation across all components
all_stds = [np.sqrt(np.diag(np.array(C))) for C in cov_list]
avg_std = np.mean(all_stds)
print(f"Average std: {avg_std:.4f}")

# distance / avg_std for every pair
for i in range(len(mu_list)):
    for j in range(i + 1, len(mu_list)):
        d = np.linalg.norm(np.array(mu_list[i]) - np.array(mu_list[j]))
        print(f"mu_{i+1} to mu_{j+1}: distance = {d:.3f}, ratio = {d / avg_std:.2f}")


L = (rng.random(10000) < .3).astype(int)   # 1 with prob 0.65, else 0
n1 = np.sum(L == 1)

n0 = np.sum(L == 0)
L = (rng.random(n0) < (.3/.7)).astype(int)
n2 = np.sum(L == 1)
n3 = np.sum(L == 0)

dist_1 = rng.multivariate_normal(dist_1_mu, dist_1_cov, size=n1)
dist_2 = rng.multivariate_normal(dist_2_mu, dist_2_cov, size=n2)
dist_3 = rng.multivariate_normal(dist_3_mu, dist_3_cov, size=int(n3/2))
dist_4 = rng.multivariate_normal(dist_4_mu, dist_4_cov, size=int(n3/2)+1)

X_dist = np.vstack([dist_1, dist_2, dist_3, dist_4])
L = np.concatenate([np.full(n1, 1), np.full(n2, 2), np.full(n3, 3)])
# Verification

print(f"Dist: X_dist | Shape: {X_dist.shape}")
print(f"Mean: {np.mean(X_dist)} | {np.mean(X_dist[L == 1], axis=0)} | {np.mean(X_dist[L == 2], axis=0)} | {np.mean(X_dist[L == 3], axis=0)}")
print(f"Labels: {np.mean(L)} | Shape: {L.shape}")



print(f"Counts-- N1: {n1} | N2: {n2} | N3: {n3}")

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(projection='3d')

markers = {1: 'o', 2: '^', 3: 's'}
colors  = {1: 'tab:blue', 2: 'tab:orange', 3: 'tab:green'}

for k in [1, 2, 3]:
    pts = X_dist[L == k]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
               marker=markers[k], c=colors[k], s=5, alpha=0.4,
               label=f"Class {k}")

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("Samples by true class label")
ax.legend()
plt.show()