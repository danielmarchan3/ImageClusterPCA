#!/usr/bin/env python
"""
Image Cluster PCA

This program performs clustering on 2D image data and organizes images into classes
based on similarity using PCA, t-SNE, and multiple clustering algorithms.

Usage:
    ./clustering_images_pca.py -i <input_images> -o <output_directory> [-m MIN_CLUSTERS] [-M MAX_CLUSTERS]
                                [-j CORES] [-p PLOTS] [-dp DEBUG_PLOTS]

Example:
    ./clustering_images_pca.py -i input.mrcs -o results_dir -m 3 -M 10 -j 8 -p 1 -dp 0

Options:
    -i, --input          Path to the input .mrcs file containing images (required).
    -o, --output         Directory to save results (required).
    -m, --min-clusters   Minimum number of clusters (default: 10).
    -M, --max-clusters   Maximum number of clusters (default: 30).
    -j, --cores          Number of CPU cores for parallel processing (default: 8).
    -p, --plots          Whether to generate plots (1 = True, 0 = False, default: 1).
    -dp, --debug-plots   Whether to generate debug plots (1 = True, 0 = False, default: 0).
    -h, --help           Show this help message and exit.
"""

import argparse

# Import custom utility functions
from utils.io import *
from utils.image_processing import *
from utils.plotting import *

# Constants
LABELS = 'labels'
SCORE = 'score'


def main(input_images, output_directory, min_clusters=10, max_clusters=30, target_size=(64, 64), cores=8, plots=1, debug_plots=0):
    """
    Main function for clustering and analyzing 2D images.

    Args:
        input_images (str): Path to input .mrcs file containing images.
        output_directory (str): Directory to save results.
        min_clusters (int): Minimum number of clusters.
        max_clusters (int): Maximum number of clusters.
        target_size (tuple): Target size to resize images.
        cores (int): Number of CPU cores for parallel processing.
        plots (int): Whether to generate plots (1 = True, 0 = False).
        debug_plots (int): Whether to generate debug plots (1 = True, 0 = False).
    """
    # Create the output directory if it doesn't exist
    create_directory(output_directory)

    # Load and preprocess images
    ref_ids_fn = input_images.replace('.mrcs', '.txt')
    image_stack_mrc, image_names =  load_images_from_mrcs(input_images, ref_ids_fn)
    image_list, image_names =  preprocess_image_stack(image_stack_mrc, image_names, target_size)

    # Build similarity matrix and generate distance matrix
    similarity_matrix = build_similarity_matrix(image_list, cpu_numbers=cores)
    distance_matrix = generate_distance_matrix(similarity_matrix)

    # Apply PCA to reduce dimensionality
    vectors, pca_explained_variance_ratio = apply_pca(distance_matrix, variance=0.95)
    cumulative_variance = pca_explained_variance_ratio.cumsum()
    n_components_95 = next(i for i, cum_var in enumerate(cumulative_variance) if cum_var >= 0.95) + 1
    print(f"Number of components to explain 95% variance: {n_components_95}")

    # Perform t-SNE for visualization
    tsne_result = apply_tsne_2d(data=vectors)

    # Adjust max_clusters if necessary
    if max_clusters == -1:
        max_clusters = len(image_list) - 2

    # Convert images and names to arrays
    image_array = np.array(image_list)
    image_names_array = np.array(image_names)

    # Determine optimal number of clusters
    optimal_clusters = determine_optimal_clusters(vectors, min_clusters=min_clusters, max_clusters=max_clusters)
    optimal_clusters_kmeans = optimal_clusters['kmeans']
    optimal_clusters_hierarchical = optimal_clusters['hierarchical']

    # Perform K-Means clustering
    labels_KMeans = perform_kmeans_clustering(data=vectors, n_clusters=optimal_clusters_kmeans)
    silhouette_kmeans = silhouette_score(vectors, labels_KMeans)

    # Perform Hierarchical clustering
    labels_hierarchical = perform_hierarchical_clustering(data=vectors, n_clusters=optimal_clusters_hierarchical)
    silhouette_hierarchical = silhouette_score(vectors, labels_hierarchical)

    # Select the best clustering method
    results = {
        'kmeans': {'labels': labels_KMeans, 'score': silhouette_kmeans},
        'hierarchical': {'labels': labels_hierarchical, 'score': silhouette_hierarchical}
    }
    best_method = max(results, key=lambda method: results[method]['score'])
    best_labels = results[best_method]['labels']
    print(f"Best method: {best_method} with Silhouette Score: {results[best_method]['score']}")

    # Save results to a file
    with open(os.path.join(output_directory, 'best_results.txt'), 'w') as f:
        f.write(f"Best clustering method: {best_method}\n")
        f.write(f"Silhouette Score: {results[best_method]['score']}\n")

    # Align images to cluster representatives and organize results
    result_dict = get_images_to_representative_alignment(best_labels, image_array, image_names_array, vectors, cores)
    labels_array, images_array, vectors_array = extract_array_like_results(result_dict)

    # Final t-SNE visualization
    final_tsne_result = apply_tsne_2d(data=vectors_array)

    # Sort images in clusters and write results
    best_clusters_with_names, sorted_results = sort_images_in_cluster(best_labels, result_dict, output_directory)

    # Plotting
    dim = target_size[0]
    zoom = 0.35 if dim > 64 else 0.7

    if plots:
        # Scatter plot of clustered images
        image_scatter_plot(final_tsne_result, images_array, labels_array, output_directory, zoom=zoom,
                           title='Aligned visualization of best-clustered images', cluster_type='best')

        # Plot all clusters in a single figure
        plot_all_clusters(sorted_results, output_directory, max_images_per_cluster=8)

    if debug_plots:
        # Debugging plots
        plot_similarity_matrix(similarity_matrix,
                               labels=[f'Image {i + 1}' for i in range(len(image_list))],
                               output_directory=output_directory)
        plot_PCA(cumulative_variance, output_directory)

        # t-SNE plots for each clustering method
        image_scatter_plot(tsne_result, image_array, labels_KMeans, output_directory,
                           title='t-SNE with K-Means Clustering', cluster_type='kMeans', zoom=zoom)
        image_scatter_plot(tsne_result, image_array, labels_hierarchical, output_directory,
                           title='t-SNE with Hierarchical Clustering', cluster_type='hierarchical', zoom=zoom)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform image clustering using PCA and t-SNE.")
    parser.add_argument('-i', '--input', required=True, help="Path to the input .mrcs file.")
    parser.add_argument('-o', '--output', required=True, help="Output directory to save results.")
    parser.add_argument('-m', '--min-clusters', type=int, default=10, help="Minimum number of clusters (default: 10).")
    parser.add_argument('-M', '--max-clusters', type=int, default=30, help="Maximum number of clusters (default: 30).")
    parser.add_argument('-j', '--cores', type=int, default=8, help="Number of CPU cores to use (default: 8).")
    parser.add_argument('-p', '--plots', type=int, default=1, help="Generate plots (1 = Yes, 0 = No, default: 1).")
    parser.add_argument('-dp', '--debug-plots', type=int, default=0, help="Generate debug plots (1 = Yes, 0 = No, default: 0).")
    args = parser.parse_args()

    main(args.input, args.output, args.min_clusters, args.max_clusters, target_size=(64, 64), cores=args.cores, plots=args.plots, debug_plots=args.debug_plots)
