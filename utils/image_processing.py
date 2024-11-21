# utils/image_processing.py
from multiprocessing import Pool, cpu_count
import numpy as np
import os
from scipy.ndimage import gaussian_filter, zoom, rotate, shift, affine_transform
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, pairwise_distances_argmin_min
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity as ssim



def gaussian_blur(image, sigma=1):
    """
    Apply Gaussian blur to the image using SciPy's gaussian_filter.

    :param image: Input image (2D or 3D array).
    :param sigma: Standard deviation for Gaussian kernel. Can be a single value or a sequence for each axis.
    :return: Blurred image.
    """
    return gaussian_filter(image, sigma=sigma)


def z_normalize(image):
    return (image - np.mean(image)) / np.std(image)


def downsample(image, target_size=(128, 128)):
    """
    Downsample the grayscale image using SciPy's zoom to resize the image to the target size.

    :param image: Input grayscale image (2D array).
    :param target_size: Target size as a tuple (height, width).
    :return: Resized grayscale image.
    """
    # Calculate zoom factors for each dimension
    zoom_factors = [target_size[0] / image.shape[0], target_size[1] / image.shape[1]]

    # Apply zoom (resizing)
    return zoom(image, zoom_factors, order=3)  # Cubic interpolation (order=3)


def preprocess_image(image, target_size=(128, 128), apply_gaussian=True):
    """
    Preprocess the input image by normalizing and downsampling.
    """

    if apply_gaussian:
        blurred_image = gaussian_blur(image)
    else:
        blurred_image = image

    downsampled_image = downsample(blurred_image, target_size)
    normalized_image = z_normalize(downsampled_image)

    return normalized_image


def preprocess_image_stack(img_stack, image_names, target_size=(128, 128)):
    images = []
    text_ids = True if image_names else False
    # Iterate over each 2D image in the stack
    for idx, img in enumerate(img_stack):
        # Preprocess the image (resize, normalize, etc.)
        preprocessed_image = preprocess_image(img, target_size=target_size)
        # Append the preprocessed image to the list
        images.append(preprocessed_image)

        if not text_ids:
            # Create a name for each image based on index or a naming convention
            filename = f"image_{idx + 1}"  # You can modify this as needed
            image_names.append(filename)

    return images, image_names


def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=img1.max() - img1.min())


def align_images(img1, img2, angle_range=180, coarse_angle_step=20, fine_angle_step=4, shift_range=12, coarse_shift_step=3, fine_shift_step=1):
    best_ssim = -1
    best_angle = 0
    best_shift = (0, 0)

    # Coarse search
    for angle in range(-angle_range, angle_range, coarse_angle_step):
        rotated_img2 = rotate(img2, angle, reshape=False)
        for x_shift in range(-shift_range, shift_range, coarse_shift_step):
            for y_shift in range(-shift_range, shift_range, coarse_shift_step):
                shifted_img2 = shift(rotated_img2, (x_shift, y_shift))
                current_ssim = compute_ssim(img1, shifted_img2)
                if current_ssim > best_ssim:
                    best_ssim = current_ssim
                    best_angle = angle
                    best_shift = (x_shift, y_shift)

    # Fine search around the best coarse result
    for angle in range(best_angle - coarse_angle_step, best_angle + coarse_angle_step + 1, fine_angle_step):
        rotated_img2 = rotate(img2, angle, reshape=False)
        for x_shift in range(best_shift[0] - coarse_shift_step, best_shift[0] + coarse_shift_step + 1, fine_shift_step):
            for y_shift in range(best_shift[1] - coarse_shift_step, best_shift[1] + coarse_shift_step + 1, fine_shift_step):
                shifted_img2 = shift(rotated_img2, (x_shift, y_shift))
                current_ssim = compute_ssim(img1, shifted_img2)
                if current_ssim > best_ssim:
                    best_ssim = current_ssim
                    best_angle = angle
                    best_shift = (x_shift, y_shift)

    aligned_img2 = shift(rotate(img2, best_angle, reshape=False), best_shift)
    transform_params = [best_angle, best_shift]

    # Create transformation matrices (Convention is different that is why the -)
    rotation_matrix = np.array([[np.cos(np.radians(-best_angle)), -np.sin(np.radians(-best_angle)), 0],
                                [np.sin(np.radians(-best_angle)), np.cos(np.radians(-best_angle)), 0],
                                [0, 0, 1]])
    translation_matrix = np.array([[1, 0, best_shift[0]],
                                   [0, 1, -best_shift[1]],
                                   [0, 0, 1]])

    # Combined transformation matrix
    combined_matrix = np.dot(translation_matrix, rotation_matrix)

    return aligned_img2, best_ssim, combined_matrix, transform_params


def transform_image(img, combined_matrix):
    '''
    This function applies a transformation matrix to an image. The transformation matrix is generated during
    alignment and can be used to align particles of different classes in a cluster to compute a new 2D average.
    This function adds a small offset.
    :param img: The image to be transformed.
    :param combined_matrix: The combined rotation and translation matrix.
    :return: The transformed image.
    '''
    # Extract rotation part and translation part from the combined matrix
    rotation_matrix = combined_matrix[:2, :2]
    translation_vector = combined_matrix[:2, 2]

    # Calculate center of the image
    center = np.array(img.shape) / 2
    offset = center - np.dot(rotation_matrix, center) + translation_vector

    # Apply affine transformation with the combined matrix
    transformed_img = affine_transform(img, rotation_matrix, offset=offset, order=1, mode='constant', cval=0.0,
                                       output_shape=img.shape)

    return transformed_img


def transform_image_shift_rotate(img, transform_params):
    '''
    This function applies a transformation functions to an image. The transformation is generated during
    alignment and can be used to align particles of different classes in a cluster to compute a new 2D average (Use this one).
    :param img:
    :param transform_params:
    :return:
    '''
    # Extract rotation part and translation part from the combined matrix
    angle = transform_params[0]
    shifts = transform_params[1]
    transformed_img = shift(rotate(img, angle, reshape=False), shifts)

    return transformed_img


def calculate_ssim_for_pair(args):
    i, j, images, shift_range, coarse_shift_step, fine_shift_step = args
    aligned_img2, best_ssim, _, _ = align_images(images[i], images[j], shift_range=shift_range, coarse_shift_step=coarse_shift_step, fine_shift_step=fine_shift_step)
    return i, j, best_ssim


def build_similarity_matrix(images, cpu_numbers=cpu_count()):
    n = len(images)
    similarity_matrix = np.zeros((n, n))

    # Compute shift parameters once
    dim = np.shape(images[0])[0]
    shift_range = int(dim * 0.20)
    coarse_shift_step = int(shift_range * 0.25)
    fine_shift_step = int(coarse_shift_step * 0.35)

    indices = [(i, j, images, shift_range, coarse_shift_step, fine_shift_step) for i in range(n) for j in range(i, n)]

    with Pool(cpu_numbers) as pool:
        results = pool.map(calculate_ssim_for_pair, indices)

    for i, j, score in results:
        similarity_matrix[i][j] = score
        similarity_matrix[j][i] = score

    return similarity_matrix


def generate_distance_matrix(similarity_matrix):
    # Convert similarity matrix to distance matrix
    return 1 - similarity_matrix


def apply_pca(distance_matrix, variance=0.95):
    """This function applies PCA with % Variance retention"""
    pca = PCA(n_components=variance)
    pca_transformed = pca.fit_transform(distance_matrix)
    # Get the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    return pca_transformed, explained_variance_ratio


def apply_tsne_2d(data):
    data_size = len(data)
    if data_size <= 10:
        perplexity = 5
    elif data_size <= 20:
        perplexity = 10
    else:
        perplexity = data_size - 10

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(data)
    return tsne_result


def determine_optimal_clusters_kmeans(data, min_clusters=3, max_clusters=10):
    """Determine the optimal number of clusters using K-means clustering."""
    wcss = []
    silhouette_scores = []
    for n in range(min_clusters, max_clusters+1):
        kmeans = KMeans(n_clusters=n, init='k-means++', n_init=10, max_iter=300, random_state=42)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    optimal_clusters = np.argmax(silhouette_scores) + min_clusters

    return optimal_clusters, wcss, silhouette_scores


def determine_optimal_clusters_hierarchical(data, min_clusters=3, max_clusters=10, linkage='ward', metric='euclidean'):
    """Determine the optimal number of clusters using hierarchical clustering."""
    silhouette_scores = []
    for n in range(min_clusters, max_clusters + 1):
        hierarchical = AgglomerativeClustering(n_clusters=n, linkage=linkage, metric=metric)
        labels = hierarchical.fit_predict(data)
        silhouette_scores.append(silhouette_score(data, labels))

    optimal_clusters = np.argmax(silhouette_scores) + min_clusters

    return optimal_clusters, silhouette_scores


def determine_optimal_clusters(data, min_clusters=3, max_clusters=10):
    """Determine the optimal clustering parameters for K-means and hierarchical."""
    optimal_kmeans, wcss, silhouette_scores_kmeans = determine_optimal_clusters_kmeans(data, min_clusters, max_clusters)
    optimal_hierarchical, silhouette_scores_hierarchical = determine_optimal_clusters_hierarchical(data, min_clusters, max_clusters)

    print(f"Optimal number of clusters for K-means: {optimal_kmeans}")
    print(f"Optimal number of clusters for Hierarchical Clustering: {optimal_hierarchical}")

    return {'kmeans': optimal_kmeans,
        'hierarchical': optimal_hierarchical}


def perform_kmeans_clustering(data, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=42)
    labels = kmeans.fit_predict(data)
    return labels


def perform_hierarchical_clustering(data, n_clusters):
    agg_clustering =  AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
    labels = agg_clustering.fit_predict(data)
    return labels


def align_single_image(args):
    """Align a single image to the representative image."""
    representative_image, image = args
    aligned_image, score, _, _ = align_images(representative_image, image)
    return aligned_image, score


def align_images_to_representative(representative_image, cluster_images, cpu_numbers=cpu_count()):
    """
    Align cluster images to the representative image using multiprocessing.

    Args:
        representative_image: The representative image to align others to.
        cluster_images: List of images in the cluster to align.
        cpu_numbers: Number of threads to use for parallel processing.

    Returns:
        tuple: (aligned_images, ssim_scores)
    """
    with Pool(cpu_numbers) as pool:
        # Prepare arguments as tuples of (representative_image, image)
        args = [(representative_image, image) for image in cluster_images]

        # Parallelize the alignment process
        results = pool.map(align_single_image, args)

    # Separate aligned images and SSIM scores from results
    aligned_images, ssim_scores = zip(*results)

    return list(aligned_images), np.array(ssim_scores)


def get_images_to_representative_alignment(labels, images, image_names, vectors, cores):
    results_cluster = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_indices = np.where(labels == label)[0]
        cluster_vectors = vectors[cluster_indices]
        cluster_images = images[cluster_indices]
        cluster_image_names = image_names[cluster_indices]
        # Compute centroid of the cluster
        centroid = np.mean(cluster_vectors, axis=0)
        # Find the image closest to the centroid
        closest, _ = pairwise_distances_argmin_min([centroid], cluster_vectors)
        representative_image = cluster_images[closest[0]]

        # Align images to the representative image
        aligned_cluster_images, ssim_scores = align_images_to_representative(representative_image, cluster_images, cores)
        results_cluster[label] = [aligned_cluster_images, cluster_image_names, ssim_scores, cluster_vectors]

    return results_cluster


def sort_images_in_cluster(labels, result_dict, output_dir):
    """
    Sorts images within each cluster based on SSIM scores and saves cluster assignments.

    Args:
        labels: List or array of cluster labels.
        result_dict: Dictionary containing aligned images, image names, SSIM scores, etc. for each cluster.
        output_dir: Directory to save the cluster assignments.

    Returns:
        best_clusters_with_names: Dictionary mapping cluster labels to sorted image names.
        sorted_results: Dictionary mapping cluster labels to sorted image data (images, names, and scores).
    """
    best_clusters_with_names = {}
    sorted_results = {}
    unique_labels = np.unique(labels)

    for label in unique_labels:
        aligned_images, image_names, ssim_scores, _ = result_dict[label]
        aligned_images = np.array(aligned_images)

        # Sort images by SSIM scores in descending order
        sorted_indices = np.argsort(ssim_scores)[::-1]
        sorted_images = aligned_images[sorted_indices]
        sorted_image_names = image_names[sorted_indices].tolist()
        sorted_ssim_values = np.array(ssim_scores)[sorted_indices]

        # Store sorted names for this cluster
        if label not in best_clusters_with_names:
            best_clusters_with_names[label] = []
        best_clusters_with_names[label].extend(sorted_image_names)

        # Store sorted data for plotting
        sorted_results[label] = (sorted_images, sorted_image_names, sorted_ssim_values)

    # Save cluster assignments to a file
    with open(os.path.join(output_dir, "best_clusters_with_names.txt"), "w") as f:
        for label, names in best_clusters_with_names.items():
            f.write(f"Cluster {label}:\n")
            for name in names:
                f.write(f"\t{name}\n")

    print("Cluster assignments with image names:", best_clusters_with_names)
    return best_clusters_with_names, sorted_results


def extract_array_like_results(result_dict):
    labels = []
    images = []
    vectors = []

    for label, results in result_dict.items():   # results_cluster[label] = [aligned_cluster_images, cluster_image_names, ssim_scores, cluster_vectors]
        aligned_cluster_images, _, _, cluster_vectors = results
        labels.extend([label] * len(cluster_vectors))
        images.extend(aligned_cluster_images)
        vectors.extend(cluster_vectors)

    labels_array = np.array(labels)
    images_array = np.array(images)
    vectors_array = np.array(vectors)

    return labels_array, images_array, vectors_array