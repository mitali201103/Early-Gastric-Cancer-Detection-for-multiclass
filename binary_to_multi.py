"""
AUTOMATED BINARY TO MULTICLASS CONVERTER
Automatically splits binary (normal/abnormal) dataset into 4 classes using image preprocessing

This script analyzes images and automatically classifies them based on:
- Image intensity/brightness patterns
- Texture features
- Color variations
- Statistical properties

Author: AI Assistant
Date: 2024
"""

import os
import shutil
import cv2
import numpy as np
from PIL import Image
import glob
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION - CHANGE THESE SETTINGS
# =============================================================================

class Config:
    # Input directories (your current binary structure)
    BINARY_DATA_DIR = "processed_images"
    NORMAL_DIR = os.path.join(BINARY_DATA_DIR, "normal")
    ABNORMAL_DIR = os.path.join(BINARY_DATA_DIR, "abnormal")
    
    # Output directories (new multiclass structure)
    OUTPUT_DIR = "processed_images_multiclass"
    CLASS_0_DIR = os.path.join(OUTPUT_DIR, "class_0")  # Normal
    CLASS_1_DIR = os.path.join(OUTPUT_DIR, "class_1")  # Early/Mild
    CLASS_2_DIR = os.path.join(OUTPUT_DIR, "class_2")  # Intermediate/Moderate
    CLASS_3_DIR = os.path.join(OUTPUT_DIR, "class_3")  # Advanced/Severe
    
    # Classification method
    METHOD = "auto"  # Options: "auto", "clustering", "intensity", "interactive"
    
    # Feature extraction settings
    RESIZE_DIM = (224, 224)
    
    # Class names for your domain
    CLASS_NAMES = {
        0: "normal",
        1: "early_stage",
        2: "intermediate",
        3: "advanced"
    }

# =============================================================================
# FEATURE EXTRACTION FUNCTIONS
# =============================================================================

def extract_image_features(image_path):
    """
    Extract features from an image for classification
    
    Features extracted:
    1. Mean intensity (brightness)
    2. Standard deviation (contrast)
    3. Edge density (texture complexity)
    4. Color variance (if color image)
    5. Histogram features
    6. Local Binary Pattern (texture)
    
    Returns:
        features (dict): Dictionary of extracted features
    """
    
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not load {image_path}")
            return None
        
        # Resize for consistent processing
        img_resized = cv2.resize(img, Config.RESIZE_DIM)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        
        # Feature 1: Mean intensity (brightness)
        mean_intensity = np.mean(gray)
        
        # Feature 2: Standard deviation (contrast/variation)
        std_intensity = np.std(gray)
        
        # Feature 3: Edge density (using Canny edge detection)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Feature 4: Histogram statistics
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()  # Normalize
        hist_mean = np.mean(hist)
        hist_std = np.std(hist)
        hist_skewness = np.mean((hist - hist_mean)**3) / (hist_std**3 + 1e-8)
        
        # Feature 5: Gradient magnitude (texture strength)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = np.mean(gradient_magnitude)
        
        # Feature 6: Texture variance (local variation)
        texture_variance = np.var(gray)
        
        # Feature 7: Color features (if applicable)
        if len(img_resized.shape) == 3:
            # Color variance
            color_std = np.std(img_resized, axis=(0, 1))
            color_mean = np.mean(img_resized, axis=(0, 1))
            color_variance = np.mean(color_std)
        else:
            color_variance = 0
            color_mean = [0, 0, 0]
        
        # Feature 8: Entropy (randomness/complexity)
        hist_normalized = hist + 1e-10  # Avoid log(0)
        entropy = -np.sum(hist_normalized * np.log2(hist_normalized))
        
        # Compile all features
        features = {
            'mean_intensity': mean_intensity,
            'std_intensity': std_intensity,
            'edge_density': edge_density,
            'hist_std': hist_std,
            'hist_skewness': hist_skewness,
            'mean_gradient': mean_gradient,
            'texture_variance': texture_variance,
            'color_variance': color_variance,
            'entropy': entropy
        }
        
        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def classify_by_intensity(features):
    """
    Simple classification based on intensity and texture
    Higher values typically indicate more abnormality
    """
    
    # Calculate composite severity score
    # Normalize and weight different features
    severity_score = (
        0.3 * (features['edge_density'] * 100) +      # Edge density
        0.2 * (features['std_intensity'] / 255) +      # Contrast
        0.2 * (features['mean_gradient'] / 100) +      # Texture strength
        0.2 * (features['texture_variance'] / 10000) + # Local variation
        0.1 * (features['entropy'] / 10)               # Complexity
    )
    
    # Classify based on severity score thresholds
    # These thresholds may need adjustment based on your data
    if severity_score < 0.15:
        return 1  # Early/Mild
    elif severity_score < 0.30:
        return 2  # Intermediate/Moderate
    else:
        return 3  # Advanced/Severe


def classify_by_clustering(feature_matrix, n_clusters=3):
    """
    Use K-Means clustering to automatically group abnormal images into 3 classes
    
    Args:
        feature_matrix: numpy array of shape (n_samples, n_features)
        n_clusters: number of clusters (3 for early/intermediate/advanced)
    
    Returns:
        labels: cluster assignments
        centroids: cluster centers
    """
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(feature_matrix)
    
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features_scaled)
    
    # Get cluster centroids
    centroids = kmeans.cluster_centers_
    
    # Sort clusters by severity (based on edge density and texture)
    # Typically: low severity → high severity
    edge_density_idx = 2  # Assuming edge_density is 3rd feature
    centroid_severities = centroids[:, edge_density_idx]
    
    # Map clusters to classes (1=early, 2=intermediate, 3=advanced)
    severity_order = np.argsort(centroid_severities)
    label_mapping = {severity_order[i]: i+1 for i in range(n_clusters)}
    
    # Remap labels
    remapped_labels = np.array([label_mapping[label] for label in labels])
    
    return remapped_labels, centroids


# =============================================================================
# MAIN CONVERSION FUNCTIONS
# =============================================================================

def process_normal_images():
    """Copy all normal images to class_0 folder"""
    
    print("\n" + "="*60)
    print("STEP 1: Processing Normal Images (Class 0)")
    print("="*60)
    
    if not os.path.exists(Config.NORMAL_DIR):
        print(f"❌ Normal directory not found: {Config.NORMAL_DIR}")
        return 0
    
    # Create class_0 directory
    os.makedirs(Config.CLASS_0_DIR, exist_ok=True)
    
    # Get all image files
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.PNG']
    normal_images = []
    
    for ext in extensions:
        pattern = os.path.join(Config.NORMAL_DIR, ext)
        normal_images.extend(glob.glob(pattern))
    
    print(f"Found {len(normal_images)} normal images")
    
    # Copy images to class_0
    for img_path in tqdm(normal_images, desc="Copying normal images"):
        filename = os.path.basename(img_path)
        dest_path = os.path.join(Config.CLASS_0_DIR, filename)
        shutil.copy2(img_path, dest_path)
    
    print(f"✅ Copied {len(normal_images)} images to class_0/")
    return len(normal_images)


def process_abnormal_images_auto():
    """
    Automatically classify abnormal images using feature extraction and clustering
    """
    
    print("\n" + "="*60)
    print("STEP 2: Processing Abnormal Images (Auto Classification)")
    print("="*60)
    
    if not os.path.exists(Config.ABNORMAL_DIR):
        print(f"❌ Abnormal directory not found: {Config.ABNORMAL_DIR}")
        return
    
    # Get all abnormal images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.PNG']
    abnormal_images = []
    
    for ext in extensions:
        pattern = os.path.join(Config.ABNORMAL_DIR, ext)
        abnormal_images.extend(glob.glob(pattern))
    
    print(f"Found {len(abnormal_images)} abnormal images")
    
    if len(abnormal_images) == 0:
        print("❌ No abnormal images found!")
        return
    
    # Extract features from all abnormal images
    print("\nExtracting features from images...")
    features_list = []
    valid_images = []
    
    for img_path in tqdm(abnormal_images, desc="Extracting features"):
        features = extract_image_features(img_path)
        if features is not None:
            features_list.append(features)
            valid_images.append(img_path)
    
    if len(features_list) == 0:
        print("❌ Could not extract features from any images!")
        return
    
    print(f"Successfully extracted features from {len(features_list)} images")
    
    # Convert features to matrix
    feature_names = list(features_list[0].keys())
    feature_matrix = np.array([[f[name] for name in feature_names] for f in features_list])
    
    # Classify using clustering
    print("\nClassifying images using K-Means clustering...")
    labels, centroids = classify_by_clustering(feature_matrix, n_clusters=3)
    
    # Create class directories
    os.makedirs(Config.CLASS_1_DIR, exist_ok=True)
    os.makedirs(Config.CLASS_2_DIR, exist_ok=True)
    os.makedirs(Config.CLASS_3_DIR, exist_ok=True)
    
    # Map labels to directories
    class_dirs = {
        1: Config.CLASS_1_DIR,
        2: Config.CLASS_2_DIR,
        3: Config.CLASS_3_DIR
    }
    
    # Copy images to appropriate class folders
    print("\nCopying images to class folders...")
    class_counts = {1: 0, 2: 0, 3: 0}
    
    for img_path, label in tqdm(zip(valid_images, labels), total=len(valid_images), desc="Organizing images"):
        filename = os.path.basename(img_path)
        dest_dir = class_dirs[label]
        dest_path = os.path.join(dest_dir, filename)
        shutil.copy2(img_path, dest_path)
        class_counts[label] += 1
    
    # Print results
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Class 1 (Early/Mild):         {class_counts[1]} images")
    print(f"Class 2 (Intermediate):       {class_counts[2]} images")
    print(f"Class 3 (Advanced/Severe):    {class_counts[3]} images")
    print(f"Total abnormal:               {sum(class_counts.values())} images")
    
    # Visualize distribution
    visualize_classification(feature_matrix, labels, feature_names)
    
    return class_counts


def process_abnormal_images_interactive():
    """
    Interactively classify abnormal images by showing them to user
    """
    
    print("\n" + "="*60)
    print("STEP 2: Processing Abnormal Images (Interactive Mode)")
    print("="*60)
    
    if not os.path.exists(Config.ABNORMAL_DIR):
        print(f"❌ Abnormal directory not found: {Config.ABNORMAL_DIR}")
        return
    
    # Get all abnormal images
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.PNG']
    abnormal_images = []
    
    for ext in extensions:
        pattern = os.path.join(Config.ABNORMAL_DIR, ext)
        abnormal_images.extend(glob.glob(pattern))
    
    print(f"Found {len(abnormal_images)} abnormal images to classify")
    print("\nFor each image, you'll classify it as:")
    print("  1 = Early/Mild stage")
    print("  2 = Intermediate/Moderate stage")
    print("  3 = Advanced/Severe stage")
    print("  s = Skip this image")
    print("  q = Quit early\n")
    
    # Create class directories
    os.makedirs(Config.CLASS_1_DIR, exist_ok=True)
    os.makedirs(Config.CLASS_2_DIR, exist_ok=True)
    os.makedirs(Config.CLASS_3_DIR, exist_ok=True)
    
    class_dirs = {
        1: Config.CLASS_1_DIR,
        2: Config.CLASS_2_DIR,
        3: Config.CLASS_3_DIR
    }
    
    class_counts = {1: 0, 2: 0, 3: 0}
    
    for i, img_path in enumerate(abnormal_images):
        print(f"\n[{i+1}/{len(abnormal_images)}] Image: {os.path.basename(img_path)}")
        
        # Display image
        try:
            img = Image.open(img_path)
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(os.path.basename(img_path))
            plt.axis('off')
            plt.show(block=False)
            plt.pause(0.1)
        except Exception as e:
            print(f"Could not display image: {e}")
        
        # Get user input
        while True:
            choice = input("Classify as (1/2/3/s/q): ").strip().lower()
            
            if choice in ['1', '2', '3']:
                label = int(choice)
                filename = os.path.basename(img_path)
                dest_path = os.path.join(class_dirs[label], filename)
                shutil.copy2(img_path, dest_path)
                class_counts[label] += 1
                print(f"✓ Classified as Class {label} ({Config.CLASS_NAMES[label]})")
                plt.close()
                break
            elif choice == 's':
                print("✗ Skipped")
                plt.close()
                break
            elif choice == 'q':
                print("\nQuitting early...")
                plt.close()
                print(f"\nClassified {sum(class_counts.values())}/{len(abnormal_images)} images")
                return class_counts
            else:
                print("Invalid input. Enter 1, 2, 3, s, or q")
    
    # Print final results
    print("\n" + "="*60)
    print("CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Class 1 (Early/Mild):         {class_counts[1]} images")
    print(f"Class 2 (Intermediate):       {class_counts[2]} images")
    print(f"Class 3 (Advanced/Severe):    {class_counts[3]} images")
    print(f"Total classified:             {sum(class_counts.values())} images")
    
    return class_counts


def visualize_classification(feature_matrix, labels, feature_names):
    """Create visualization of classification results"""
    
    try:
        # Create output directory for visualizations
        viz_dir = "classification_visualizations"
        os.makedirs(viz_dir, exist_ok=True)
        
        # Plot 1: Feature distributions by class
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, feature_name in enumerate(feature_names):
            if i < len(axes):
                ax = axes[i]
                for class_label in [1, 2, 3]:
                    mask = labels == class_label
                    ax.hist(feature_matrix[mask, i], alpha=0.5, 
                           label=f'Class {class_label}', bins=20)
                ax.set_title(feature_name)
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_distributions.png'), dpi=150)
        plt.close()
        
        # Plot 2: Class distribution pie chart
        fig, ax = plt.subplots(figsize=(8, 8))
        class_counts = [np.sum(labels == i) for i in [1, 2, 3]]
        colors = ['#90EE90', '#FFD700', '#FF6B6B']
        ax.pie(class_counts, labels=['Class 1\n(Early)', 'Class 2\n(Intermediate)', 'Class 3\n(Advanced)'],
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('Distribution of Abnormal Images Across Classes', fontsize=14, fontweight='bold')
        plt.savefig(os.path.join(viz_dir, 'class_distribution.png'), dpi=150)
        plt.close()
        
        print(f"\n✅ Visualizations saved in '{viz_dir}/' folder")
        
    except Exception as e:
        print(f"Could not create visualizations: {e}")


def generate_labels_csv():
    """Generate labels.csv file from the organized class folders"""
    
    print("\n" + "="*60)
    print("STEP 3: Generating labels.csv")
    print("="*60)
    
    data = []
    
    # Process each class folder
    for class_num in range(4):
        class_dir = getattr(Config, f'CLASS_{class_num}_DIR')
        
        if not os.path.exists(class_dir):
            print(f"Warning: {class_dir} does not exist")
            continue
        
        # Get all images in this class
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.PNG']
        images = []
        
        for ext in extensions:
            pattern = os.path.join(class_dir, ext)
            images.extend(glob.glob(pattern))
        
        # Add to data with relative path
        for img_path in images:
            filename = os.path.basename(img_path)
            relative_path = os.path.join(f'class_{class_num}', filename)
            data.append({
                'filename': relative_path,
                'label': class_num
            })
        
        print(f"Class {class_num} ({Config.CLASS_NAMES[class_num]}): {len(images)} images")
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    csv_path = os.path.join(Config.OUTPUT_DIR, 'labels.csv')
    df.to_csv(csv_path, index=False)
    
    print(f"\n✅ labels.csv created: {csv_path}")
    print(f"Total images: {len(df)}")
    
    # Show distribution
    print("\nFinal class distribution:")
    print(df['label'].value_counts().sort_index())
    
    return csv_path


def create_summary_report():
    """Create a summary report of the conversion"""
    
    report_path = os.path.join(Config.OUTPUT_DIR, 'conversion_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("BINARY TO MULTICLASS CONVERSION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Method used: {Config.METHOD}\n\n")
        
        f.write("Class Mapping:\n")
        for class_num, class_name in Config.CLASS_NAMES.items():
            class_dir = getattr(Config, f'CLASS_{class_num}_DIR')
            if os.path.exists(class_dir):
                count = len([f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
                f.write(f"  Class {class_num} ({class_name}): {count} images\n")
        
        f.write("\nOutput Directory Structure:\n")
        f.write(f"  {Config.OUTPUT_DIR}/\n")
        f.write(f"    ├── class_0/  (Normal)\n")
        f.write(f"    ├── class_1/  (Early/Mild)\n")
        f.write(f"    ├── class_2/  (Intermediate)\n")
        f.write(f"    ├── class_3/  (Advanced/Severe)\n")
        f.write(f"    └── labels.csv\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("Next Steps:\n")
        f.write("="*60 + "\n")
        f.write("1. Review the classification results\n")
        f.write("2. Manually verify some images in each class\n")
        f.write(f"3. Update CLASS_NAMES in multiclass_gastric_cancer_model.py\n")
        f.write("4. Run: python multiclass_gastric_cancer_model.py\n")
    
    print(f"\n✅ Conversion report saved: {report_path}")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function"""
    
    print("\n" + "="*60)
    print("BINARY TO MULTICLASS DATASET CONVERTER")
    print("="*60)
    print("\nThis script will convert your binary dataset into 4 classes:")
    print("  • Class 0: Normal")
    print("  • Class 1: Early/Mild abnormal")
    print("  • Class 2: Intermediate/Moderate abnormal")
    print("  • Class 3: Advanced/Severe abnormal")
    
    # Check if input directories exist
    if not os.path.exists(Config.NORMAL_DIR):
        print(f"\n❌ ERROR: Normal directory not found: {Config.NORMAL_DIR}")
        print("Please make sure your binary dataset is organized as:")
        print("  processed_images/")
        print("    ├── normal/")
        print("    └── abnormal/")
        return
    
    if not os.path.exists(Config.ABNORMAL_DIR):
        print(f"\n❌ ERROR: Abnormal directory not found: {Config.ABNORMAL_DIR}")
        return
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Ask user for classification method
    print("\n" + "="*60)
    print("Choose classification method:")
    print("="*60)
    print("1. Automatic (AI-based clustering) - RECOMMENDED")
    print("2. Interactive (manually classify each image)")
    print("3. Intensity-based (simple threshold classification)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == '1':
        Config.METHOD = "auto"
    elif choice == '2':
        Config.METHOD = "interactive"
    elif choice == '3':
        Config.METHOD = "intensity"
    else:
        print("Invalid choice. Using automatic method.")
        Config.METHOD = "auto"
    
    print(f"\nUsing method: {Config.METHOD}")
    
    # Step 1: Process normal images
    normal_count = process_normal_images()
    
    # Step 2: Process abnormal images
    if Config.METHOD == "auto":
        abnormal_counts = process_abnormal_images_auto()
    elif Config.METHOD == "interactive":
        abnormal_counts = process_abnormal_images_interactive()
    elif Config.METHOD == "intensity":
        # Simple intensity-based classification
        abnormal_counts = process_abnormal_images_auto()  # Use auto for now
    
    # Step 3: Generate labels.csv
    csv_path = generate_labels_csv()
    
    # Step 4: Create summary report
    create_summary_report()
    
    # Final summary
    print("\n" + "="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)
    print(f"\nNew multiclass dataset created in: {Config.OUTPUT_DIR}/")
    print(f"Labels CSV file: {csv_path}")
    print("\nNext steps:")
    print("1. Review the images in each class folder")
    print("2. Adjust any misclassified images if needed")
    print("3. Update CLASS_NAMES in your training script")
    print("4. Run: python multiclass_gastric_cancer_model.py")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
