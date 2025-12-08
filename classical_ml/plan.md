Classical ML Approaches for Car Damage Segmentation
Overview
Implement 3 classical machine learning approaches to perform pixel-level segmentation on car damage images. The dataset contains COCO-format annotations with segmentation polygons, bounding boxes, and 6 damage categories (dent, scratch, crack, glass shatter, lamp broken, tire flat).

Dataset Structure
Training: dataset/train/ - 2816 images + _annotations.coco.json
Validation: dataset/valid/ - 810 images + _annotations.coco.json  
Test: dataset/test/ - 374 images + _annotations.coco.json
Format: COCO JSON with segmentation polygons, bounding boxes, and category IDs
Implementation Plan
1. Data Loading and Preprocessing Module
File: classical_ml/data_loader.py

Load COCO JSON annotations
Convert segmentation polygons to binary masks
Extract bounding boxes for patch-based approaches
Create data loaders for train/val/test splits
Handle multi-class segmentation (6 damage types + background)
2. Approach 1: Sliding Window + Traditional Classifier
File: classical_ml/sliding_window_classifier.py

Extract image patches using sliding window technique
Extract features from patches (color histograms, texture features, etc.)
Train SVM or Random Forest classifier on patches
Apply classifier to full images using sliding window
Reconstruct segmentation masks from patch predictions
Post-process to smooth boundaries
3. Approach 2: HOG + SVM
File: classical_ml/hog_svm.py

Extract Histogram of Oriented Gradients (HOG) features from image patches
Train SVM classifier on HOG features
Apply sliding window with HOG feature extraction
Classify patches and reconstruct segmentation masks
Handle multi-class classification (one-vs-rest or multi-class SVM)
4. Approach 3: Color/Texture Features + Random Forest
File: classical_ml/color_texture_rf.py

Extract pixel-level features:
Color features (RGB, HSV histograms)
Texture features (Local Binary Patterns, GLCM)
Spatial context features
Train Random Forest classifier for pixel-level classification
Generate pixel-level segmentation masks
Apply post-processing for smoothness
5. Evaluation Module
File: classical_ml/evaluation.py

Calculate metrics: IoU (Intersection over Union), pixel accuracy, Dice coefficient
Per-class and overall metrics
Compare predictions with ground truth masks
Generate confusion matrices
6. Visualization Module
File: classical_ml/visualization.py

Visualize predictions overlaid on original images
Show ground truth vs predictions side-by-side
Display segmentation masks with different colors for each damage type
Generate comparison plots for all 3 approaches
7. Main Training/Evaluation Script
File: classical_ml/main.py

Orchestrate training and evaluation of all 3 approaches
Command-line interface for running experiments
Save models and results
Generate reports comparing all approaches
8. Requirements
File: classical_ml/requirements.txt

scikit-learn (SVM, Random Forest)
scikit-image (HOG, LBP, image processing)
opencv-python (image operations)
numpy, pandas
matplotlib, seaborn (visualization)
pycocotools (COCO format handling)
Implementation Details
Feature Extraction
Sliding Window: Extract patches of size 64x64 or 128x128 with overlap
HOG: Use scikit-image's hog() function with appropriate parameters
Color/Texture: Extract features in a window around each pixel for context
Training Strategy
Balance classes (damage vs background) using sampling
Use validation set for hyperparameter tuning
Save trained models for inference
Post-processing
Apply morphological operations to smooth masks
Remove small isolated regions
Fill holes in predicted masks
Expected Outputs
Trained models for each approach
Segmentation predictions on test set
Evaluation metrics and comparison report
Visualization images showing predictions
Summary report comparing all 3 approaches