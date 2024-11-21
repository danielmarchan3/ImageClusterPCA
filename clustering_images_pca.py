#!/usr/bin/env python
import numpy as np
from sklearn.metrics import silhouette_score
import sys

from utils.io import *
from utils.image_processing import *
from utils.plotting import *


# Constants
LABELS = 'labels'
SCORE = 'score'
FN = "class_representatives"


def main(input_images, output_directory, min_clusters=3, max_clusters=10, target_size=(64, 64), cores=8, plots=1, debug_plots=0):
    """Main function to execute image clustering."""
    create_directory(output_directory)
    # Load images and preprocess
    imgs_fn = input_images  # .mrcs file
    ref_ids_fn = imgs_fn.replace('.mrcs', '.txt')

    image_list, image_names = load_and_preprocess_images_from_mrcs(imgs_fn, ref_ids_fn, target_size)

    # Build similarity matrix
    similarity_matrix = build_similarity_matrix(image_list, cpu_numbers=)

    # Generate distance matrix
    distance_matrix = generate_distance_matrix(similarity_matrix)

    # PCA to reduce the dimensionality: apply PCA with 95% Variance retention
    vectors, pca_explained_variance_ratio = apply_pca(distance_matrix, variance=0.95)
    # Calculate the cumulative explained variance
    cumulative_variance = pca_explained_variance_ratio.cumsum()
    # Determine the number of components required to explain 95% of the variance
    n_components_95 = next(i for i, cumulative_var in enumerate(cumulative_variance) if cumulative_var >= 0.95) + 1
    print(f"Number of components to explain 95% of the variance: {n_components_95}")

    # Perform t-SNE for visualization of multidimensional clustering into 2D
    tsne_result = apply_tsne_2d(data=vectors)

    if max_clusters == -1:  # Calculate the max number of clusters based on the number of references
        max_clusters = len(image_list) - 2

    image_array = np.array(image_list)
    image_names_array = np.array(image_names)

    # Determine optimal number of clusters
    optimal_clusters = determine_optimal_clusters(vectors,
                                                  min_clusters=min_clusters, max_clusters=max_clusters)

    optimal_clusters_kmeans = optimal_clusters['kmeans']
    optimal_clusters_hierarchical = optimal_clusters['hierarchical']

    # Comparing Clustering with optimal number of clusters
    # Perform K-means clustering
    kMeans_results = {}
    labels_KMeans = perform_kmeans_clustering(data=vectors, n_clusters=optimal_clusters_kmeans)
    kMeans_results[LABELS] = labels_KMeans

    # Perform Hierarchical clustering
    hierarchical_results = {}
    labels_hierarchical = perform_hierarchical_clustering(data=vectors, n_clusters=optimal_clusters_hierarchical)
    hierarchical_results[LABELS] = labels_hierarchical

    # Compute Silhouette Score for each clustering method
    silhouette_kmeans = silhouette_score(vectors, labels_KMeans)
    silhouette_hierarchical = silhouette_score(vectors, labels_hierarchical)

    print(f'Silhouette Score for K-Means: {silhouette_kmeans}')
    print(f'Silhouette Score for Hierarchical Clustering: {silhouette_hierarchical}')

    kMeans_results[SCORE] = silhouette_kmeans
    hierarchical_results[SCORE] = silhouette_hierarchical

    results = {'kmeans': kMeans_results, 'hierarchical': hierarchical_results}

    # Filter out None results
    valid_results = {method: res for method, res in results.items() if res[SCORE] is not None}

    # Find and print the best method and its Silhouette Score
    if valid_results:
        best_method = max(valid_results, key=lambda method: valid_results[method][SCORE])
        best_result = valid_results[best_method]

        print(f"Best clustering method: {best_method}")
        print(f"Best Silhouette Score: {best_result[SCORE]}")

        filename = os.path.join(output_directory, 'best_results.txt')
        with open(filename, 'w') as file:
            file.write(f"Best clustering method: {best_method}\n")
            file.write(f"Best Silhouette Score: {best_result[SCORE]}")

        # Use the best labels for further analysis or visualization
        best_labels = best_result[LABELS]
    else:
        print("No valid clustering results available.")
        exit(0)

    result_dict = get_images_to_representative_alignment(best_labels, image_array, image_names_array, vectors, cores)
    labels_array, images_array, vectors_array = extract_array_like_results(result_dict)

    # Use t-SNE to have a 2D visual representation of the clustering
    final_tsne_result = apply_tsne_2d(data=vectors_array)

    # Sort images in cluster and write results
    best_clusters_with_names, sorted_results = sort_images_in_cluster(best_labels, result_dict, output_directory)

    # Visualize t-SNE with kmeans clustering labels
    dim = target_size[0]
    zoom = 0.35 if dim > 64 else 0.7

    if plots:
        image_scatter_plot(final_tsne_result, images_array, labels_array, output_directory, zoom=zoom,
                      title='Aligned visualization of best clustered aligned images', cluster_type='best')

        plot_all_clusters(sorted_results, output_directory, max_images_per_cluster=8)
        # In case you want to have one figure per cluster
        # plot_individual_clusters(sorted_results, output_directory, max_images_per_cluster=8)

    if debug_plots:
        # Plot similarity matrix
        plot_similarity_matrix(similarity_matrix,
                               labels=[f'Image {i + 1}' for i in range(len(image_list))],
                               output_directory=output_directory)

        # plot the cumulative explained variance
        plot_PCA(cumulative_variance, output_directory)

        # Visualize t-SNE with kmeans clustering labels
        image_scatter_plot(tsne_result, image_array, labels_KMeans, output_directory,
                            title='t-SNE with K-mean Clustering', cluster_type='kMeans', zoom=zoom)
        # Visualize t-SNE with hierarchical clustering labels
        image_scatter_plot(tsne_result, image_array, labels_hierarchical, output_directory,
                            title='t-SNE with Hierarchical Clustering', cluster_type='hierarchical', zoom=zoom)


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: ./clustering_classes_pca.py <input_images> <output_directory> [min_clusters] [max_clusters] [cores] [plots] [debug_plots]")
    else:
        input_directory = sys.argv[1]
        output_directory = sys.argv[2]
        min_clusters = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        max_clusters = int(sys.argv[4]) if len(sys.argv) > 4 else 20
        cores = int(sys.argv[5]) if len(sys.argv) > 5 else 8
        plots = int(sys.argv[6]) if len(sys.argv) > 6 else 0
        debug_plots = int(sys.argv[7]) if len(sys.argv) > 7 else 0

        main(input_directory, output_directory, min_clusters, max_clusters,
             target_size=(64, 64), cores=cores, plots=plots, debug_plots=debug_plots)

    exit('Finish clustering')