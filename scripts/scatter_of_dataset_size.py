import matplotlib.pyplot as plt
import numpy as np
import os

# Data
n_samples = [50, 75, 100, 125]
auc_scores = [0.915, 0.879, 0.857, 0.851]
ap_scores = [0.947, 0.938, 0.930, 0.916]

# Custom save path
save_dir = 'pictures_1014/3/grok3-v11' # Modify to your desired path

# Ensure save directory exists
os.makedirs(save_dir, exist_ok=True)

# Calculate y-axis range
auc_min, auc_max = min(auc_scores), max(auc_scores)
ap_min, ap_max = min(ap_scores), max(ap_scores)

# Add some padding to y-axis
auc_padding = (auc_max - auc_min) * 0.15
ap_padding = (ap_max - ap_min) * 0.15

# Create and save AUC scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(n_samples, auc_scores, color='blue', s=80, alpha=0.7, edgecolors='black')
plt.xlabel('Sample Size (n_samples)', fontsize=12)
plt.ylabel('AUC Score', fontsize=12)
plt.title('Sample Size vs AUC Score', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Set y-axis range for AUC plot
plt.ylim(auc_min - auc_padding, auc_max + auc_padding)

# Add labels for each point
for i, (x, y) in enumerate(zip(n_samples, auc_scores)):
    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                 xytext=(0,10), ha='center', fontsize=9)

# Save AUC plot
auc_filename = 'nsamples_vs_auc.png'
auc_save_path = os.path.join(save_dir, auc_filename)
plt.tight_layout()
plt.savefig(auc_save_path, dpi=300, bbox_inches='tight')
plt.close()  # Close current figure, prepare for next one

print(f"AUC plot saved to: {auc_save_path}")

# Create and save AP scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(n_samples, ap_scores, color='green', s=80, alpha=0.7, edgecolors='black')
plt.xlabel('Sample Size (n_samples)', fontsize=12)
plt.ylabel('AP Score', fontsize=12)
plt.title('Sample Size vs AP Score', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

# Set y-axis range for AP plot
plt.ylim(ap_min - ap_padding, ap_max + ap_padding)

# Add labels for each point
for i, (x, y) in enumerate(zip(n_samples, ap_scores)):
    plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points",
                 xytext=(0,10), ha='center', fontsize=9)

# Save AP plot
ap_filename = 'nsamples_vs_ap.png'
ap_save_path = os.path.join(save_dir, ap_filename)
plt.tight_layout()
plt.savefig(ap_save_path, dpi=300, bbox_inches='tight')
plt.close()  # Close current figure

print(f"AP plot saved to: {ap_save_path}")

# Print statistics
print("\nAUC Scores:")
for i, (n, score) in enumerate(zip(n_samples, auc_scores)):
    print(f"  {n} samples: {score:.3f}")

print("\nAP Scores:")
for i, (n, score) in enumerate(zip(n_samples, ap_scores)):
    print(f"  {n} samples: {score:.3f}")