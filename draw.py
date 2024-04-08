import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    
    :param ax: matplotlib.axes.Axes instance
    :param left: float, left horizontal position of the figure
    :param right: float, right horizontal position of the figure
    :param bottom: float, bottom vertical position of the figure
    :param top: float, top vertical position of the figure
    :param layer_sizes: list of int, size of each layer
    '''
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = patches.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                    edgecolor='k', facecolor='w', zorder=4)
            ax.add_patch(circle)
            # Annotations for layers
            if n == 0:
                ax.annotate(f'Input\n{num_features} features', (n*h_spacing + left, layer_top - m*v_spacing),
                            textcoords="offset points", xytext=(-10,-25), ha='center', fontsize=8)
            elif n == len(layer_sizes) - 1:
                ax.annotate(f'Output\n{num_classes} classes', (n*h_spacing + left, layer_top - m*v_spacing),
                            textcoords="offset points", xytext=(-10,-20), ha='center', fontsize=8)
            else:
                ax.annotate(f'{layer_sizes[n]}', (n*h_spacing + left, layer_top - m*v_spacing),
                            textcoords="offset points", xytext=(-10,-10), ha='center', fontsize=8)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = patches.FancyArrowPatch((n*h_spacing + left, layer_top_a - m*v_spacing),
                                                ((n + 1)*h_spacing + left, layer_top_b - o*v_spacing),
                                                connectionstyle="arc3,rad=.1", arrowstyle='-', color="k", lw=0.5)
                ax.add_patch(line)

# Network architecture
num_features = 4 # Example feature size, to be adjusted based on actual input features
num_classes = 3 # Example class size, adjust based on actual classes
layer_sizes = [num_features, 10, 5, 2, num_classes]  # Layers in the network

fig = plt.figure(figsize=(10, 6))
ax = fig.gca()
ax.axis('off')
draw_neural_net(ax, .1, .9, .1, .9, layer_sizes)
plt.title("Fishing Vessel Neural Network Architecture")
plt.show()