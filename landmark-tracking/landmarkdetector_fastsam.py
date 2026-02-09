#!/usr/bin/env python3
"""
Landmark Detection - BehAV Navigation System

Detects landmarks in images using FastSAM segmentation + GPT-4 Vision.
Compares a ground truth image of a known landmark with a test image to:
1. Determine if the landmark is present
2. Identify which mask contains the landmark
3. Estimate the camera distance from the landmark

Usage:
    # Show help
    python landmarkdetector_fastsam.py --help

    # Run demo with default Testudo images
    python landmarkdetector_fastsam.py --demo

    # Specify custom images
    python landmarkdetector_fastsam.py -g Images/Iribe/Ground_truth.jpg -t Images/Iribe/1.jpg

    # Specify reference distance (default 3m for Testudo)
    python landmarkdetector_fastsam.py --demo --ref-distance 80

Docker:
    docker-compose --profile test run --rm landmark_test --demo
    docker-compose --profile test run --rm landmark_test -g Images/Testudo/Ground_truth.webp -t Images/Testudo/1.jpg

Environment:
    OPENAI_API_KEY - Required. Your OpenAI API key for GPT-4 Vision.
    FASTSAM_MODEL_PATH - Optional. Path to FastSAM-x.pt model (default: /app/models/FastSAM-x.pt)

Available test image sets:
    - Testudo (statue, ref: 3m)
    - Iribe (building, ref: 80m)
    - Douglass (building, ref: 15m)
    - M_circle (landmark, ref: 21.38m)
    - Idea_factory (building, ref: 29.1m)
    - Chapel (building, ref: 32m)
"""

import requests
import base64
import time
import numpy as np
import cv2
import io
import os
import sys
import re
import argparse
from PIL import Image
from skimage.measure import regionprops
from skimage import measure
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
from requests.exceptions import RequestException
import torch

# Lazy load FastSAM to allow --help without GPU
FastSAM = None
FastSAMPrompt = None

def load_fastsam():
    """Lazy load FastSAM modules."""
    global FastSAM, FastSAMPrompt
    if FastSAM is None:
        print(f"CUDA available: {torch.cuda.is_available()}")
        from FastSAM.fastsam.model import FastSAM as _FastSAM
        from FastSAM.fastsam.prompt import FastSAMPrompt as _FastSAMPrompt
        FastSAM = _FastSAM
        FastSAMPrompt = _FastSAMPrompt


def get_openai_api_key():
    """Get OpenAI API key from environment variable."""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with: export OPENAI_API_KEY=your-key-here"
        )
    return api_key


# Default paths - can be overridden via environment variables
DEFAULT_FASTSAM_MODEL_PATH = os.environ.get('FASTSAM_MODEL_PATH', '/app/models/FastSAM-x.pt')
DEFAULT_SAVE_IMAGE_PLOT = os.environ.get('SAVE_IMAGE_PLOT_DIR', './Image_plots/')


class LandmarkDetector:
    def __init__(self, api_key=None, ground_truth_image_path=None, test_image_path=None,
                 fastsam_model_path=None, save_image_plot_dir=None, ref_distance=3.0, verbose=False):
        # Get API key from parameter or environment
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = get_openai_api_key()

        self.ground_truth_image_path = ground_truth_image_path
        self.test_image_path = test_image_path
        self.image_plot = True
        self.save_image_plot = save_image_plot_dir or DEFAULT_SAVE_IMAGE_PLOT
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ref_distance = ref_distance
        self.verbose = verbose

        # Lazy load FastSAM
        load_fastsam()

        # Load FastSAM model from parameterized path
        model_path = fastsam_model_path or DEFAULT_FASTSAM_MODEL_PATH
        print(f"Loading FastSAM model from: {model_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"FastSAM model not found at: {model_path}")
        self.model = FastSAM(model_path)
        self.max_retries = 3
        self.delay = 5

    def get_next_file_number(self):
        files = os.listdir(self.save_image_plot)
        existing_numbers = []
        for f in files:
            if f.startswith('Image_plots_') and f.endswith('.jpg'):
                match = re.search(r'Image_plots_(\d+)\.jpg', f)
                if match:
                    existing_numbers.append(int(match.group(1)))
        return max(existing_numbers, default=0) + 1

    def save_images(self,original_img, masked_img, circled_img, mask_number, distance):
        os.makedirs(self.save_image_plot, exist_ok=True)
        # os.makedirs(self.base_path_mask, exist_ok=True)
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(original_img)
        plt.title('Original image')
        plt.axis('off')
        plt.subplot(132)
        plt.imshow(masked_img)
        plt.title('Masked image')
        plt.axis('off')
        plt.subplot(133)
        plt.imshow(circled_img)
        plt.title('Circled image')
        plt.axis('off')
        plt.tight_layout()
        next_number = self.get_next_file_number()
        img_plot_filename = f"Image_plots_{next_number}.jpg" 
        self.save_path_plot = os.path.join(self.save_image_plot, img_plot_filename)
        plt.figtext(0.06, 0.95, f"Mask Number: {mask_number}, \nDistance: {distance}", 
                    ha="center", va = "top", fontsize=12, bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        plt.savefig(self.save_path_plot)
        plt.close()

    def make_api_request(self, headers, data):
        for attempt in range(self.max_retries):
            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers= headers, json=data, timeout=30)
                response.raise_for_status()
                return response.json()
            except RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt + 1 < self.max_retries:
                    print(f"Retrying in {self.delay} seconds...")
                    time.sleep(self.delay)
                else:
                    print("Max retries reached. Unable to get a response.")
                    return None
                
    def load_image_path(self, file_path):
        with open(file_path, "rb") as image_file:
            return f"data:image/jpeg;base64,{base64.b64encode(image_file.read()).decode('utf-8')}"
    
    def load_image(self, image):
        pil_image = Image.fromarray(image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"


    def get_mask_coordinates(self, masks, target_mask_number):
        if target_mask_number < 1 or target_mask_number > len(masks):
            raise ValueError(f"Invalid mask number: {target_mask_number}. Should be between 1 and {len(masks)}")
        else:
            # print(target_mask_number)
            mask = masks[target_mask_number - 1]
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        binary_mask = (mask > 0).astype(int)
        regions = regionprops(binary_mask)
        if len(regions) == 0:
            raise ValueError("No regions found in the binary mask. The mask might be empty.")
        for i, region in enumerate(regions):
            centroid = region.centroid  # (y, x) coordinates
            bbox = region.bbox  # (min_row, min_col, max_row, max_col)
            
            # Calculate the center of the bounding box
            center_y = (bbox[0] + bbox[2]) // 2
            center_x = (bbox[1] + bbox[3]) // 2
        region = regions[0]
        centroid = region.centroid
        bbox = region.bbox
        center_y = (bbox[0] + bbox[2]) // 2
        center_x = (bbox[1] + bbox[3]) // 2
        return (center_x, center_y)   

    def plot_dot_on_image(self, image, x, y, radius=40, color=(255, 0, 0), thickness=-1):
        # if isinstance(image, str):
        #     image = cv2.imread(image)
        if not isinstance(image, np.ndarray):
            raise ValueError("The 'image' argument must be a numpy array or a valid file path.")
        circled_img = image.copy()
        cv2.circle(circled_img, (int(x), int(y)), radius, color, thickness)
        return circled_img

    def process_and_display_masks(self, masks, image, distance_threshold=200):
        """Merge nearby masks and number them on the image.

        Returns (annotated_image, merged_mask_arrays) where merged_mask_arrays
        is a list of binary numpy arrays suitable for get_mask_coordinates().
        """
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        merged_masks = []
        used_masks = set()

        for i in range(len(masks)):
            if i in used_masks:
                continue
            merged_mask = masks[i].astype(bool).copy()
            for j in range(i + 1, len(masks)):
                if j in used_masks:
                    continue
                other_mask = masks[j].astype(bool)
                if np.any(binary_dilation(merged_mask, iterations=distance_threshold) & other_mask):
                    merged_mask |= other_mask
                    used_masks.add(j)
            merged_masks.append(merged_mask)

        if len(merged_masks) == 0:
            return image, []

        # Sort by area (largest first)
        merged_masks.sort(key=lambda m: np.sum(m), reverse=True)

        # Draw contours
        annotated = image.copy()
        for mask in merged_masks:
            contours = measure.find_contours(mask, 0.5)
            for contour in contours:
                contour = np.array(contour, dtype=np.int32)
                cv2.polylines(annotated, [contour[:, [1, 0]]], isClosed=True,
                              color=(255, 255, 255), thickness=2)

        # Compute centroids and sort by position (top-to-bottom, left-to-right)
        props = []
        for idx, mask in enumerate(merged_masks):
            y_coords, x_coords = np.where(mask)
            cx, cy = np.mean(x_coords), np.mean(y_coords)
            props.append((idx, cx, cy))
        props.sort(key=lambda p: (p[2], p[1]))

        # Number each mask
        ordered_masks = []
        for number, (idx, cx, cy) in enumerate(props, start=1):
            cv2.putText(annotated, str(number), (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
            ordered_masks.append(merged_masks[idx])

        return annotated, ordered_masks

    def run(self):
        start_time = time.time()
        image_feed = cv2.imread(self.test_image_path)
        image = image_feed
        image = cv2.cvtColor(image_feed, cv2.COLOR_BGR2RGB)

        results = self.model(image, device=self.device, retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(image, results, device=self.device)
        ann = prompt_process.everything_prompt()
        masked_image, merged_ann = self.process_and_display_masks(ann, image)
        prompt_presence = f"""Compare two images:
            1. A ground truth image of a landmark taken from {self.ref_distance} meters away.
            2. A masked test image with numbered segments.

            Task 1: Is the landmark present in the test image?
            Example Response format for Task 1:
            Landmark - YES/NO

            If YES, proceed to Tasks 2 and 3:

            Task 2: Identify the mask number containing the landmark (even partially).
            Task 3: Estimate the camera distance from the landmark in the test image.
            Use the ground truth image ({self.ref_distance}m reference) to estimate relative size.

            Example response format for Task 2 and Task 3:
            Mask number: 3
            Distance: 120 meters """

        # Reference distances for different landmarks:
        # Iribe - 80 meters
        # Douglass - 15 meters
        # M_circle - 21.38 meters
        # Idea_factory - 29.1 meters
        # Testudo - 3 meters
        # Chapel - 32 meters
        
        headers = {
        'Authorization': f'Bearer {self.api_key}',
        'Content-Type': 'application/json'
        }
        data_presence = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_presence},
                        {"type": "image_url", "image_url": {"url": self.load_image_path(self.ground_truth_image_path)}},
                        {"type": "image_url", "image_url": {"url": self.load_image(masked_image)}}
                    ]
                }
            ],
            "max_tokens": 300
        }
        response_json = self.make_api_request(headers, data_presence)

        if self.verbose and response_json:
            print(f"\n[VERBOSE] Raw GPT-4 Vision response:")
            print(response_json['choices'][0]['message']['content'].strip())
            print()

        end_time = time.time()
        elapsed_time = end_time - start_time

        if response_json:
            response_text = response_json['choices'][0]['message']['content'].strip()
            try:
                landmark_status = response_text.split('-')[1].strip().split('\n')[0].upper()
            except (IndexError, AttributeError):
                landmark_status = "NO"

            if landmark_status == 'YES':
                # Extract mask number
                mask_number_match = re.search(r'Mask number:\s*(\d+)', response_text)
                distance_match = re.search(r'Distance:\s*(\d+(?:\.\d+)?)\s*(?:meters?|m)?', response_text, re.IGNORECASE)

                mask_number = int(mask_number_match.group(1)) if mask_number_match else None
                distance_str = distance_match.group(0) if distance_match else None

                if mask_number:
                    try:
                        coordinates = self.get_mask_coordinates(merged_ann, mask_number)
                        X, Y = coordinates[0], coordinates[1]
                        circled_img = self.plot_dot_on_image(image, X, Y)

                        if self.image_plot:
                            self.save_images(image, masked_image, circled_img, mask_number, distance_str or "Unknown")
                            print(f"\nOutput saved to: {self.save_path_plot}")

                        print_results(
                            landmark_present=True,
                            mask_number=mask_number,
                            distance=distance_str,
                            pixel_coords=(X, Y),
                            elapsed_time=elapsed_time
                        )
                    except (ValueError, IndexError) as e:
                        print(f"Error extracting mask coordinates: {e}")
                        print_results(landmark_present=True, mask_number=mask_number,
                                    distance=distance_str, elapsed_time=elapsed_time)
                else:
                    print("Warning: Landmark detected but mask number not found in response")
                    print_results(landmark_present=True, elapsed_time=elapsed_time)
            else:
                print_results(landmark_present=False, elapsed_time=elapsed_time)
        else:
            print("Error: Unable to get response from GPT-4 Vision API")
            print_results(landmark_present=False, elapsed_time=elapsed_time)


def print_results(landmark_present, mask_number=None, distance=None, pixel_coords=None, elapsed_time=None):
    """Print formatted detection results."""
    separator = "=" * 70

    print(f"\n{separator}")
    print("LANDMARK DETECTION RESULTS")
    print(separator)

    print(f"\n  Landmark Present: {'YES' if landmark_present else 'NO'}")

    if landmark_present and mask_number:
        print(f"\n{'-' * 70}")
        print("LOCALIZATION")
        print(f"{'-' * 70}")
        print(f"\n  Mask Number:      {mask_number}")
        if distance:
            print(f"  Estimated Distance: {distance}")
        if pixel_coords:
            print(f"  Pixel Location:   X={pixel_coords[0]}, Y={pixel_coords[1]}")

    if elapsed_time:
        print(f"\n  Processing Time:  {elapsed_time:.2f} seconds")

    print(f"\n{separator}\n")


# Demo image sets with their reference distances
DEMO_SETS = {
    'Testudo': {'gt': 'Images/Testudo/Ground_truth.webp', 'test': 'Images/Testudo/1.jpg', 'ref': 3.0},
    'Iribe': {'gt': 'Images/Iribe/2.jpg', 'test': 'Images/Iribe/1.jpg', 'ref': 80.0},
    'Douglass': {'gt': 'Images/Douglass/2.jpg', 'test': 'Images/Douglass/1.jpg', 'ref': 15.0},
    'M_circle': {'gt': 'Images/M_circle/2.jpg', 'test': 'Images/M_circle/1.jpg', 'ref': 21.38},
    'Idea_factory': {'gt': 'Images/Idea_factory/2.jpg', 'test': 'Images/Idea_factory/1.jpg', 'ref': 29.1},
    'Chapel': {'gt': 'Images/Chapel/2.jpg', 'test': 'Images/Chapel/1.jpg', 'ref': 32.0},
}


def main():
    parser = argparse.ArgumentParser(
        description="Detect landmarks in images using FastSAM + GPT-4 Vision.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo                    # Run with Testudo demo images
  %(prog)s --demo --landmark Iribe   # Run with Iribe building
  %(prog)s -g gt.jpg -t test.jpg     # Use custom images
  %(prog)s -g gt.jpg -t test.jpg --ref-distance 50

Available demo landmarks: Testudo, Iribe, Douglass, M_circle, Idea_factory, Chapel

Environment Variables:
  OPENAI_API_KEY       Required. Your OpenAI API key.
  FASTSAM_MODEL_PATH   Optional. Path to FastSAM-x.pt model.

Docker Usage:
  docker-compose --profile test run --rm landmark_test --demo
  docker-compose --profile test run --rm landmark_test -g Images/Testudo/Ground_truth.webp -t Images/Testudo/1.jpg
        """
    )

    parser.add_argument(
        '-g', '--ground-truth',
        dest='ground_truth',
        help='Path to ground truth image of the landmark'
    )
    parser.add_argument(
        '-t', '--test-image',
        dest='test_image',
        help='Path to test image to search for landmark'
    )
    parser.add_argument(
        '--ref-distance',
        type=float,
        default=3.0,
        help='Reference distance (meters) at which ground truth was taken (default: 3.0)'
    )
    parser.add_argument(
        '--demo', '-d',
        action='store_true',
        help='Run demo with predefined test images'
    )
    parser.add_argument(
        '--landmark', '-l',
        choices=list(DEMO_SETS.keys()),
        default='Testudo',
        help='Which landmark to use for demo (default: Testudo)'
    )
    parser.add_argument(
        '--model',
        dest='model_path',
        help='Path to FastSAM model (default: from FASTSAM_MODEL_PATH env or /app/models/FastSAM-x.pt)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        dest='output_dir',
        default='./Image_plots/',
        help='Directory to save output plots (default: ./Image_plots/)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show verbose output including raw API responses'
    )

    args = parser.parse_args()

    # Determine image paths
    if args.demo:
        demo_set = DEMO_SETS[args.landmark]
        # Adjust paths for Docker environment
        base_path = '/app/landmark-tracking/' if os.path.exists('/app/landmark-tracking') else ''
        ground_truth_path = os.path.join(base_path, demo_set['gt'])
        test_image_path = os.path.join(base_path, demo_set['test'])
        ref_distance = demo_set['ref']

        print(f"\nRunning demo with {args.landmark} landmark")
        print(f"  Ground truth: {ground_truth_path}")
        print(f"  Test image:   {test_image_path}")
        print(f"  Ref distance: {ref_distance}m")
    elif args.ground_truth and args.test_image:
        ground_truth_path = args.ground_truth
        test_image_path = args.test_image
        ref_distance = args.ref_distance
    else:
        parser.print_help()
        print("\n" + "-" * 70)
        print("Error: Provide --demo or both -g (ground truth) and -t (test image)")
        sys.exit(1)

    # Verify images exist
    for img_path, desc in [(ground_truth_path, "Ground truth"), (test_image_path, "Test image")]:
        if not os.path.exists(img_path):
            print(f"Error: {desc} not found: {img_path}")
            sys.exit(1)

    # Run detection
    try:
        detector = LandmarkDetector(
            ground_truth_image_path=ground_truth_path,
            test_image_path=test_image_path,
            fastsam_model_path=args.model_path,
            save_image_plot_dir=args.output_dir,
            ref_distance=ref_distance,
            verbose=args.verbose
        )
        detector.run()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

