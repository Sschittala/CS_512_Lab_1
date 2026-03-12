import numpy as np
import matplotlib.pyplot as plt
from q2_gradient import load_train, load_model, forward_backward


# Compute KL divergence between two probability distributions
def kl_divergence(p, q):
   eps = 1e-12
   p = np.clip(p, eps, 1)
   q = np.clip(q, eps, 1)
   return np.sum(p * np.log(p / q))


# Compute total KL divergence for node marginals across all positions
def node_kl(p_true, p_est):
   m = p_true.shape[0]
   kl = 0
   for s in range(m):
       kl += kl_divergence(p_est[s], p_true[s])
   return kl


# Compute total KL divergence for edge marginals across all positions
def edge_kl(p_true, p_est):
   m = p_true.shape[0]
   kl = 0
   for s in range(m):
       kl += kl_divergence(p_est[s].ravel(), p_true[s].ravel())
   return kl


# Gibbs sampling to estimate node and edge marginals
def gibbs_sample(X, W, T, S):
   m = X.shape[0]
   K = W.shape[0]
   node_counts = np.zeros((m, K))
   edge_counts = np.zeros((m-1, K, K))
   y = np.random.randint(K, size=m)
   for t in range(S):
       for s in range(m):
           scores = W @ X[s]
           if s > 0:
               scores += T[y[s-1], :]
           if s < m-1:
               scores += T[:, y[s+1]]
          
           scores -= np.max(scores)
           probs = np.exp(scores)
           probs /= np.sum(probs)
           y[s] = np.random.choice(K, p=probs)
       for s in range(m):
           node_counts[s, y[s]] += 1
       for s in range(m-1):
           edge_counts[s, y[s], y[s+1]] += 1
   node_marg = node_counts / np.sum(node_counts, axis=1, keepdims=True)
   edge_marg = edge_counts / np.sum(edge_counts, axis=(1,2), keepdims=True)
   return node_marg, edge_marg


# Gibbs sampling with Rao-Blackwellization to reduce variance
def gibbs_sample_rb(X, W, T, S):
   m = X.shape[0]
   K = W.shape[0]
   node_counts = np.zeros((m, K))
   edge_counts = np.zeros((m-1, K, K))
   y = np.random.randint(K, size=m)
   for t in range(S):
       for s in range(m):
           scores = W @ X[s]
           if s > 0:
               scores += T[y[s-1], :]
           if s < m-1:
               scores += T[:, y[s+1]]
          
           scores -= np.max(scores)
           probs = np.exp(scores)
           probs /= np.sum(probs)
           # Rao-Blackwellization: accumulate expected probabilities
           node_counts[s] += probs
           if s > 0:
               for c in range(K):
                   edge_counts[s-1, y[s-1], c] += probs[c]
           if s < m-1:
               for c in range(K):
                   edge_counts[s, c, y[s+1]] += probs[c]
           # Sample new label
           y[s] = np.random.choice(K, p=probs)
   node_marg = node_counts / np.sum(node_counts, axis=1, keepdims=True)
   edge_marg = edge_counts / np.sum(edge_counts, axis=(1,2), keepdims=True)
   return node_marg, edge_marg


# Main function for Q4c
def run_q4c():
   W, T = load_model("../data/model.txt")
   train_x, train_y = load_train("../data/train.txt")
   X = train_x[0]
   y = train_y[0]
   _, true_node, true_edge = forward_backward(X, W, T, y)
   samples = np.unique(np.round(np.logspace(0, 3, num=10))).astype(int)
   node_plain = []
   edge_plain = []
   node_rb = []
   edge_rb = []
   for S in samples:
       # Standard Gibbs sampling
       node_est, edge_est = gibbs_sample(X, W, T, S)
       node_plain.append(node_kl(true_node, node_est))
       edge_plain.append(edge_kl(true_edge, edge_est))
      
       # Rao-Blackwellized Gibbs sampling
       node_est_rb, edge_est_rb = gibbs_sample_rb(X, W, T, S)
       node_rb.append(node_kl(true_node, node_est_rb))
       edge_rb.append(edge_kl(true_edge, edge_est_rb))
      
       print("Samples:", S)
      
   plt.figure(figsize=(7,5))
   plt.plot(samples, node_plain, label="Node MCMC")
   plt.plot(samples, edge_plain, label="Edge MCMC")
   plt.plot(samples, node_rb, label="Node Rao-Blackwell")
   plt.plot(samples, edge_rb, label="Edge Rao-Blackwell")
   plt.xscale("log")
   plt.yscale("log")
   plt.xlabel("Number of samples")
   plt.ylabel("KL divergence")
   plt.title("Q4c: KL divergence vs samples")
   plt.legend()
   plt.grid(True)
   plt.savefig("result/q4c_plot.png")
   plt.show()
  
run_q4c()