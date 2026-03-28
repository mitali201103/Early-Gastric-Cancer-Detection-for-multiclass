import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
from skimage import exposure
from skimage.color import rgb2hed, hed2rgb
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# STEP 1: Configure your paths
INPUT_DIR = "raw_images"           
OUTPUT_DIR = "processed_images"    
TARGET_SIZE = (512, 512)           

def create_output_folders():
    """Create necessary output folders"""
    folders = ['processed', 'quality_filtered', 'augmented', 'patches']
    for folder in folders:
        os.makedirs(os.path.join(OUTPUT_DIR, folder), exist_ok=True)
    print("✓ Output folders created")

def assess_image_quality(image):
    """Check if image is good quality (not blurry, not too dark/bright)"""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    brightness = np.mean(gray)
    
    contrast = np.std(gray)
    
    return {
        'sharpness': laplacian_var,
        'brightness': brightness,
        'contrast': contrast,
        'is_good_quality': laplacian_var > 50 and 30 < brightness < 200 and contrast > 20
    }

def normalize_staining(image):
    """Standardize H&E stain colors"""
    try:
        hed = rgb2hed(image)
        
        hed_normalized = exposure.rescale_intensity(hed, out_range=(0, 1))

        rgb_normalized = hed2rgb(hed_normalized)

        return (np.clip(rgb_normalized, 0, 1) * 255).astype(np.uint8)
    except:
        return image

def augment_image(image, filename):
    """Create additional training data through augmentation"""
    augmented = []

    for angle in [90, 180, 270]:
        rotated = np.rot90(image, k=angle//90)
        augmented.append((f"{filename}_rot{angle}", rotated))

    h_flip = cv2.flip(image, 1)
    v_flip = cv2.flip(image, 0)
    augmented.append((f"{filename}_hflip", h_flip))
    augmented.append((f"{filename}_vflip", v_flip))

    pil_img = Image.fromarray(image)
    bright = np.array(ImageEnhance.Brightness(pil_img).enhance(1.2))
    dim = np.array(ImageEnhance.Brightness(pil_img).enhance(0.8))
    augmented.append((f"{filename}_bright", bright))
    augmented.append((f"{filename}_dim", dim))
    
    return augmented

def extract_patches(image, filename, patch_size=256):
    """Extract smaller patches from large images"""
    h, w = image.shape[:2]
    patches = []

    n_patches_h = h // patch_size
    n_patches_w = w // patch_size
    
    patch_count = 0
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y1 = i * patch_size
            y2 = y1 + patch_size
            x1 = j * patch_size
            x2 = x1 + patch_size
            
            patch = image[y1:y2, x1:x2]

            gray_patch = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            tissue_percentage = np.sum(gray_patch > 20) / (patch_size * patch_size)
            
            if tissue_percentage > 0.3:  # At least 30% tissue
                patches.append((f"{filename}_patch_{patch_count}", patch))
                patch_count += 1
    
    return patches

def process_single_image(image_path, do_augmentation=True, extract_patches_flag=False):
    """Process one image through the complete pipeline - NO BACKGROUND REMOVAL"""
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    try:
        # Step 1: Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load {filename}")
            return False
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Step 2: Check quality
        quality = assess_image_quality(image)
        
        if not quality['is_good_quality']:
            print(f"⚠️  Skipping {filename} - Low quality (sharpness: {quality['sharpness']:.1f})")
            return False
        
        # Step 3: Resize to standard size
        image = cv2.resize(image, TARGET_SIZE)
        
        # Step 4: Normalize staining (removed background removal step)
        image = normalize_staining(image)
        
        # Step 5: Save processed image
        processed_path = os.path.join(OUTPUT_DIR, 'processed', f"{filename}_processed.png")
        cv2.imwrite(processed_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Step 6: Data augmentation (optional)
        if do_augmentation:
            augmented_images = augment_image(image, filename)
            for aug_name, aug_img in augmented_images:
                aug_path = os.path.join(OUTPUT_DIR, 'augmented', f"{aug_name}.png")
                cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
        
        # Step 7: Extract patches (optional)
        if extract_patches_flag:
            patches = extract_patches(image, filename)
            for patch_name, patch_img in patches:
                patch_path = os.path.join(OUTPUT_DIR, 'patches', f"{patch_name}.png")
                cv2.imwrite(patch_path, cv2.cvtColor(patch_img, cv2.COLOR_RGB2BGR))
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {filename}: {str(e)}")
        return False

def visualize_processing_steps(sample_image_path):
    """Show before/after comparison - Fixed version without background removal"""
    try:
        print("Creating processing visualization...")

        original = cv2.imread(sample_image_path)
        if original is None:
            print("Could not load sample image for visualization")
            return
            
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        original = cv2.resize(original, TARGET_SIZE)

        normalized = normalize_staining(original)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(normalized)
        axes[1].set_title('Stain Normalized')
        axes[1].axis('off')
        
        plt.tight_layout()
        

        output_path = os.path.join(OUTPUT_DIR, 'processing_comparison.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"Visualization saved to: {output_path}")
        
    except Exception as e:
        print(f"Visualization failed: {str(e)}")
        print("But your image processing was successful!")

def main():
    """Main processing function - Fixed version without background removal"""
    print("🔬 Gastric Cancer Image Preprocessing Pipeline")
    print("=" * 50)
    
    # Step 1: Create folders
    create_output_folders()
    
    # Step 2: Find all images
    supported_formats = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp')
    image_files = []
    
    for file in os.listdir(INPUT_DIR):
        if file.lower().endswith(supported_formats):
            image_files.append(os.path.join(INPUT_DIR, file))
    
    print(f"📁 Found {len(image_files)} images to process")
    
    if len(image_files) == 0:
        print("❌ No images found! Make sure your images are in the 'raw_images' folder")
        return
    
    # Step 3: Process all images
    successful = 0
    quality_data = []
    
    print("\n🔄 Processing images...")
    for image_path in tqdm(image_files, desc="Processing"):
        success = process_single_image(
            image_path, 
            do_augmentation=True,      
            extract_patches_flag=False  
        )
        
        if success:
            successful += 1
    
    print(f"\n✅ Successfully processed {successful}/{len(image_files)} images")

    processed_count = len(os.listdir(os.path.join(OUTPUT_DIR, 'processed')))
    augmented_count = len(os.listdir(os.path.join(OUTPUT_DIR, 'augmented')))
    
    print(f"📊 Results:")
    print(f"   - Processed images: {processed_count}")
    print(f"   - Augmented images: {augmented_count}")
    print(f"   - Total dataset size: {processed_count + augmented_count}")
    
    if image_files:
        print(f"\n📈 Creating processing visualization...")
        try:
            visualize_processing_steps(image_files[0])
        except Exception as e:
            print(f"Visualization skipped: {str(e)} (but processing was successful)")
    
    print(f"\n🎉 Preprocessing complete! Check the '{OUTPUT_DIR}' folder for results.")

if __name__ == "__main__":
    main()