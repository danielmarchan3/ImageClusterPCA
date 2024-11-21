# utils/plotting.py
import os
import numpy as np



def plot_similarity_matrix(similarity_matrix, labels=None, output_directory=''):
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.title('Image Similarity Matrix (SSIM)')
    plt.xlabel('Image Index')
    plt.ylabel('Image Index')
    plot_path = os.path.join(output_directory, 'similarity_matrix.png')
    plt.savefig(plot_path)
    plt.close()


def plot_PCA(cumulative_variance, output_directory):
    import matplotlib.pyplot as plt
    # Optionally, plot the cumulative explained variance
    plt.figure(figsize=(8, 5))
    plt.plot(cumulative_variance, marker='o')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance explained')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance by PCA Components')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_directory, 'pca_cumulative_components.png'))
    plt.close()


def image_scatter_plot(vectors, images, labels, output_directory, zoom=0.35,
                       title='t-SNE visualization of clustered images', cluster_type=''):
    """
    Create a scatter plot of images using t-SNE results.
    Displays the images in grayscale with distinct border colors representing their cluster labels.
    Includes a legend for cluster colors.

    Args:
        vectors: 2D array-like (t-SNE or PCA coordinates for images).
        images: List of image arrays (grayscale images expected).
        labels: Array of cluster labels for coloring borders.
        output_directory: Directory to save the resulting plot.
        zoom: Float value to control the size of images in the scatter plot.
        title: Title of the plot.
        cluster_type: Name of the cluster type (used in the output filename).
    """
    import matplotlib.pyplot as plt
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(16, 10))

    # Get unique cluster labels and their count
    unique_labels = np.unique(labels)
    num_clusters = len(unique_labels)

    # Logic for selecting the colormap
    if num_clusters <= 10:
        cmap = plt.get_cmap('tab10')  # Up to 10 clusters
    elif num_clusters <= 20:
        cmap = plt.get_cmap('tab20')  # Between 11 and 20 clusters
    else:
        # For more than 20 clusters, generate a custom colormap using HSV
        colors = plt.cm.hsv(np.linspace(0, 1, num_clusters))  # Generate distinct colors
        cmap = ListedColormap(colors)

    # Assign distinct colors to each cluster
    cluster_colors = {label: cmap(i / num_clusters) for i, label in enumerate(unique_labels)}

    # Scatter plot (no colormap normalization, uses cluster_colors)
    scatter = ax.scatter(vectors[:, 0], vectors[:, 1], c=[cluster_colors[label] for label in labels], alpha=0.6)

    # Annotate the plot with images
    for xy, img, label in zip(vectors, images, labels):
        # Create the OffsetImage with the 'gray' colormap
        imagebox = OffsetImage(img, cmap='gray', zoom=zoom)
        imagebox.image.axes = ax
        # Set the border color according to the assigned cluster color
        ab = AnnotationBbox(imagebox, xy, frameon=True,
                            bboxprops=dict(edgecolor=cluster_colors[label], lw=2))
        ax.add_artist(ab)

    # Add a legend for cluster labels and their colors
    legend_elements = [
        Patch(facecolor=color, edgecolor=color, label=f'Cluster {label}')
        for label, color in cluster_colors.items()
    ]
    ax.legend(handles=legend_elements, title="Clusters", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

    # Finalize the plot
    plt.title(title)
    tsne_plot_path = os.path.join(output_directory, cluster_type + '_cluster_visualization_with_images.png')
    plt.savefig(tsne_plot_path, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved to {tsne_plot_path}")


def plot_individual_clusters(sorted_results, output_dir, max_images_per_cluster=5):
    """
    Creates separate plots for each cluster.

    Args:
        sorted_results: Dictionary mapping cluster labels to sorted image data (images, names, and scores).
        output_dir: Directory to save the plots.
        max_images_per_cluster: Maximum number of images to display per cluster.
    """
    import matplotlib.pyplot as plt

    for label, (sorted_images, _, sorted_ssim_values) in sorted_results.items():
        num_images = min(len(sorted_images), max_images_per_cluster)

        plt.figure(figsize=(num_images * 2, 4))
        for i in range(num_images):
            plt.subplot(1, num_images, i + 1)
            plt.imshow(sorted_images[i], cmap='gray')
            plt.axis('off')
            plt.title(f"SSIM: {sorted_ssim_values[i]:.2f}", fontsize=8)

        plt.suptitle(f"Cluster {label}", fontsize=12)
        plt.tight_layout()

        # Save the plot
        output_file = os.path.join(output_dir, f"cluster_{label}.png")
        plt.savefig(output_file)
        plt.close()


def plot_all_clusters(sorted_results, output_dir, max_images_per_cluster=8):
    """
    Plots all clusters and their images in a single plot with cluster names and counts displayed vertically.

    Args:
        sorted_results: Dictionary mapping cluster labels to sorted image data (images, names, and scores).
        output_dir: Directory to save the consolidated plot.
        max_images_per_cluster: Maximum number of images to display per cluster.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    num_clusters = len(sorted_results)

    # Define figure and GridSpec
    fig = plt.figure(figsize=(max_images_per_cluster * 2, num_clusters * 2))
    gs = gridspec.GridSpec(num_clusters, max_images_per_cluster + 1, width_ratios=[0.5] + [1] * max_images_per_cluster)

    for i, (label, (sorted_images, _, sorted_ssim_values)) in enumerate(sorted_results.items()):
        # Get the number of images in the current cluster
        num_images_in_cluster = len(sorted_images)

        # Add vertical cluster name and image count to the first column
        ax_label = fig.add_subplot(gs[i, 0])
        ax_label.axis('off')  # Turn off axis
        ax_label.text(
            0.5, 0.5,
            f"$\\bf{{Cluster\ {label}}}$\n({num_images_in_cluster} images)",  # Bold cluster name
            fontsize=12, ha='center', va='center', rotation=90
        )

        # Add images to subsequent columns
        num_images = min(num_images_in_cluster, max_images_per_cluster)
        for j in range(max_images_per_cluster):
            ax = fig.add_subplot(gs[i, j + 1])
            if j < num_images:
                ax.imshow(sorted_images[j], cmap='gray')  # Display image in grayscale
                ax.set_title(f"SSIM: {sorted_ssim_values[j]:.2f}", fontsize=8)
            else:
                ax.axis('off')  # Turn off empty image slots
            ax.axis('off')

    # Adjust layout for better appearance
    plt.tight_layout()

    # Save the plot
    output_file = os.path.join(output_dir, "all_clusters_with_labels.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Consolidated cluster plot with labels and counts saved to {output_file}")