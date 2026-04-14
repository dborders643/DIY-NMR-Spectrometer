import numpy as np
import matplotlib.pyplot as plt

def get_magnet_nodes(radius, thickness, z_offset):
    # Returns the 4 unique corners for FEMM (Axisymmetric: r, z)
    # Ordered: Bottom-Left, Bottom-Right, Top-Right, Top-Left
    nodes = np.array([
        [0, z_offset],              # Node 1
        [radius, z_offset],         # Node 2
        [radius, z_offset + thickness], # Node 3
        [0, z_offset + thickness]   # Node 4
    ])
    return nodes

# --- Parameters ---
rad = 1.0       # Radius
thick = 0.375   # Thickness (3/8")
gap = 1.0       # Distance between magnets

# Calculate Node Sets
# Bottom magnet sits below the gap
bot_nodes = get_magnet_nodes(rad, thick, z_offset = -gap/2 - thick)
# Top magnet sits above the gap
top_nodes = get_magnet_nodes(rad, thick, z_offset = gap/2)

all_sets = [("Bottom", bot_nodes, 'blue'), ("Top", top_nodes, 'red')]

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 10))

for name, nodes, color in all_sets:
    # Draw the rectangle
    rect = np.vstack([nodes, nodes[0]])
    ax.plot(rect[:, 0], rect[:, 1], color=color, lw=2, label=f'{name} Magnet')
    
    # Label each of the 4 nodes per magnet
    for i, (r, z) in enumerate(nodes):
        ax.scatter(r, z, color='black', zorder=5)
        # Offset the text slightly for readability
        ax.text(r + 0.05, z + 0.02, f'({r:.3f}, {z:.3f})', 
                fontsize=9, fontweight='bold', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

# Aesthetics
ax.axhline(0, color='black', lw=1, ls='--') # Gap center
ax.axvline(0, color='black', lw=1)          # Symmetry axis
ax.set_xlabel('Radius (r)')
ax.set_ylabel('Elevation (z)')
ax.set_title(f'FEMM Node Mapping (Gap={gap}, Thickness={thick})')
ax.grid(True, which='both', linestyle=':', alpha=0.5)
ax.axis('equal')
ax.legend()

plt.show()

# --- Console Output for FEMM Transfer ---
print("FEMM COORDINATE LIST (r, z):")
print("-" * 30)
for name, nodes, _ in all_sets:
    print(f"\n{name} Magnet Nodes:")
    for i, pt in enumerate(nodes):
        print(f"  Node {i+1}: r={pt[0]:.4f}, z={pt[1]:.4f}")