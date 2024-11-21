# Clustering Images PCA

This program performs clustering on 2D images and organizes images into classes based on similarity using PCA, t-SNE, and multiple clustering algorithms.

## Features:
- Image preprocessing
- Clustering and dimensionality reduction using PCA
- Image alignment with SSIM scoring
- Cluster visualization with t-SNE

## Requirements:
You can set up the environment using the following Conda command:

    conda env create -f environment.yml

Alternatively, you can use `pip` with:

    pip install -r requirements.txt

## Usage:
Once the environment is set up, run the program with:

    python clustering_images_pca.py -i <input_images> -o <output_directory> [-m MIN_CLUSTERS] [-M MAX_CLUSTERS] [-j CORES] [-p PLOTS] [-dp DEBUG_PLOTS]

### Example:
    python clustering_images_pca.py -i input.mrcs -o results_dir -m 3 -M 10 -j 8 -p 1 -dp 0

Options:

    -i, --input          Path to the input .mrcs file containing images (required). \n
    -o, --output         Directory to save results (required).
    -m, --min-clusters   Minimum number of clusters (default: 10).
    -M, --max-clusters   Maximum number of clusters (default: 30).
    -j, --cores          Number of CPU cores for parallel processing (default: 8).
    -p, --plots          Whether to generate plots (1 = True, 0 = False, default: 1).
    -dp, --debug-plots   Whether to generate debug plots (1 = True, 0 = False, default: 0).
    -h, --help           Show this help message and exit.

This will execute the clustering process and produce visualizations of the clusters.

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.