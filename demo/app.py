#!/usr/bin/env python3
"""
Gradio Demo App for Patchioner Model - Trace-based Image Captioning

This demo allows users to:
1. Upload or select an image
2. Draw traces on the image using Gradio's ImageEditor
3. Generate captions for the traced regions using a pre-trained Patchioner model

Author: Generated for decap-dino project
"""

import sys
import os
# Add the project root directory to Python path to import src modules
sys.path.append(os.path.join( os.path.dirname(os.path.abspath(__file__)), 'Patch-ioner'))

import gradio as gr

from gradio_image_annotation import image_annotator as foo_image_annotator

import torch
import yaml
import json
import traceback
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Dict, Optional

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '../Patch-ioner', '.env'))

# Import the Patchioner model from the src directory
from src.model import Patchioner

# Global variable to store the loaded model
loaded_model = None
model_config_path = None
device = "cuda" if torch.cuda.is_available() else "cpu"

# Default model configuration
DEFAULT_MODEL_CONFIG = "mlp.k.yaml"

# Example images directory
current_dir = os.path.dirname(__file__)
EXAMPLE_IMAGES_DIR = Path(os.path.join(current_dir, 'example-images')).resolve()
CONFIGS_DIR = Path(os.path.join(current_dir, '../Patch-ioner/configs')).resolve()


def initialize_default_model() -> str:
    """Initialize the default model at startup."""
    global loaded_model, model_config_path
    
    try:
        # Look for the default config file
        default_config_path = CONFIGS_DIR / DEFAULT_MODEL_CONFIG
        
        if not default_config_path.exists():
            return f"❌ Default config file not found: {default_config_path}"
        
        print(f"Loading default model: {DEFAULT_MODEL_CONFIG}")
        
        # Load and parse the config
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load the model using the from_config class method
        model = Patchioner.from_config(config, device=device)
        model.eval()
        model.to(device)
        
        # Store the model globally
        loaded_model = model
        model_config_path = str(default_config_path)
        
        return f"✅ Default model loaded: {DEFAULT_MODEL_CONFIG} on {device}"
        
    except Exception as e:
        error_msg = f"❌ Error loading default model: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg


def get_example_images(limit=None) -> List[str]:
    """Get list of example images for the demo."""
    example_images = []
    if EXAMPLE_IMAGES_DIR.exists():
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            example_images.extend(str(p) for p in EXAMPLE_IMAGES_DIR.glob(ext))
    if limit is not None:
        example_images = example_images[:limit]
    return example_images
    

def get_example_configs() -> List[str]:
    """Get list of example config files."""
    example_configs = []
    if CONFIGS_DIR.exists():
        example_configs = [str(p) for p in CONFIGS_DIR.glob("*.yaml")]
    else:
        print(f"Warning: Configs directory {CONFIGS_DIR} does not exist.")
    return sorted(example_configs)


def load_model_from_config(config_file_path: str) -> str:
    """
    Load the Patchioner model from a config file.
    
    Args:
        config_file_path: Path to the YAML configuration file
        
    Returns:
        Status message about model loading
    """
    global loaded_model, model_config_path
    
    try:
        if not config_file_path or not os.path.exists(config_file_path):
            return "❌ Error: Config file path is empty or file does not exist."
        
        print(f"Loading model from config: {config_file_path}")
        
        # Load and parse the config
        with open(config_file_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load the model using the from_config class method
        model = Patchioner.from_config(config, device=device)
        model.eval()
        model.to(device)
        
        # Store the model globally
        loaded_model = model
        model_config_path = config_file_path
        
        return f"✅ Model loaded successfully from {os.path.basename(config_file_path)} on {device}"
        
    except Exception as e:
        error_msg = f"❌ Error loading model: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg


def process_image_trace_to_coordinates(image_editor_data) -> List[List[Dict[str, float]]]:
    """
    Convert Gradio ImageEditor trace data to the coordinate format expected by the model.
    
    The expected format is: [[{"x": float, "y": float, "t": float}, ...], ...]
    where coordinates are normalized to [0, 1] and t is a timestamp.
    
    Args:
        image_editor_data: Data from Gradio ImageEditor component
        
    Returns:
        List of traces in the expected format
    """
    try:
        print(f"[DEBUG] process_image_trace_to_coordinates called")
        print(f"[DEBUG] image_editor_data type: {type(image_editor_data)}")
        
        if image_editor_data is None:
            print("[DEBUG] image_editor_data is None")
            return []
            
        if isinstance(image_editor_data, dict):
            print(f"[DEBUG] Available keys in image_editor_data: {list(image_editor_data.keys())}")
        
        # Check for different possible structures
        layers = None
        if isinstance(image_editor_data, dict):
            if 'layers' in image_editor_data:
                layers = image_editor_data['layers']
            elif 'composite' in image_editor_data:
                # Sometimes gradio stores drawing data differently
                composite = image_editor_data['composite']
                if isinstance(composite, dict) and 'layers' in composite:
                    layers = composite['layers']
        
        if not layers:
            print("[DEBUG] No layers found in image_editor_data")
            return []
        
        traces = []
        print(f"[DEBUG] Processing {len(layers)} layers")
        
        # Process each drawing layer - they are PIL Images, not coordinate data
        for i, layer in enumerate(layers):
            print(f"[DEBUG] Processing layer {i}: {layer}")
            
            # Skip if layer is not a PIL Image or is empty
            if not isinstance(layer, Image.Image):
                print(f"[DEBUG] Layer {i} is not a PIL Image")
                continue
            
            # Convert layer to numpy array to find non-transparent pixels
            layer_array = np.array(layer)
            
            # Find non-transparent pixels (alpha > 0)
            if layer_array.shape[2] == 4:  # RGBA
                non_transparent = layer_array[:, :, 3] > 0
            else:  # RGB - assume any non-black pixel is drawn
                non_transparent = np.any(layer_array > 0, axis=2)
            
            # Get coordinates of drawn pixels
            y_coords, x_coords = np.where(non_transparent)
            
            if len(x_coords) == 0:
                print(f"[DEBUG] Layer {i} has no drawn pixels")
                continue
            
            print(f"[DEBUG] Layer {i} has {len(x_coords)} drawn pixels")
            
            # Convert pixel coordinates to trace format
            trace_points = []
            img_height, img_width = layer_array.shape[:2]
            
            # Sample some points from the drawn pixels (to avoid too many points)
            num_points = min(len(x_coords), 100)  # Limit to 100 points max
            if num_points > 0:
                # Sample evenly spaced indices
                indices = np.linspace(0, len(x_coords) - 1, num_points, dtype=int)
                sampled_x = x_coords[indices]
                sampled_y = y_coords[indices]
                
                # Convert to normalized coordinates and create trace points
                for idx, (x, y) in enumerate(zip(sampled_x, sampled_y)):
                    # Normalize coordinates to [0, 1]
                    x_norm = float(x) / img_width if img_width > 0 else 0
                    y_norm = float(y) / img_height if img_height > 0 else 0
                    
                    # Clamp to [0, 1] range
                    x_norm = max(0, min(1, x_norm))
                    y_norm = max(0, min(1, y_norm))
                    
                    # Add timestamp (arbitrary progression)
                    t = idx * 0.1
                    
                    trace_points.append({
                        "x": x_norm,
                        "y": y_norm,
                        "t": t
                    })
            
            if trace_points:
                traces.append(trace_points)
        
        return traces
        
    except Exception as e:
        print(f"Error processing image trace: {e}")
        print(traceback.format_exc())
        return []


def process_bounding_box_coordinates(annotator_data) -> List[List[float]]:
    """
    Convert Gradio image_annotator data to bounding box format expected by the model.
    
    Args:
        annotator_data: Data from Gradio image_annotator component
        
    Returns:
        List of bounding boxes in [x, y, width, height] format
    """
    try:
        print(f"[DEBUG] process_bounding_box_coordinates called")
        print(f"[DEBUG] annotator_data type: {type(annotator_data)}")
        #print(f"[DEBUG] annotator_data content: {annotator_data}")
        
        if annotator_data is None:
            print("[DEBUG] annotator_data is None")
            return []
            
        boxes = []
        
        # Handle the dictionary format from image_annotator
        if isinstance(annotator_data, dict):
            print(f"[DEBUG] Available keys in annotator_data: {list(annotator_data.keys())}")
            
            # Extract boxes from the 'boxes' key
            if 'boxes' in annotator_data and annotator_data['boxes']:
                for box in annotator_data['boxes']:
                    if isinstance(box, dict):
                        # Based on image_annotator.py, boxes have format:
                        # {"xmin": x, "ymin": y, "xmax": x2, "ymax": y2, "label": ..., "color": ...}
                        xmin = box.get('xmin', 0)
                        ymin = box.get('ymin', 0)
                        xmax = box.get('xmax', 0)
                        ymax = box.get('ymax', 0)
                        
                        width = xmax - xmin
                        height = ymax - ymin
                        
                        # Convert to [x, y, width, height] format
                        boxes.append([xmin, ymin, width, height])
            else:
                print("[DEBUG] No 'boxes' key found or boxes list is empty")
        
        print(f"[DEBUG] Found {len(boxes)} bounding boxes: {boxes}")
        return boxes
        
    except Exception as e:
        print(f"Error processing bounding box: {e}")
        print(traceback.format_exc())
        return []


def generate_caption(mode, image_data) -> str:
    """
    Generate caption for the image and traces/bboxes using the loaded model.
    
    Args:
        mode: Either "trace" or "bbox" mode
        image_data: Data from Gradio ImageEditor or Annotate component
        
    Returns:
        Generated caption or error message
    """
    global loaded_model
    
    try:
        print(f"[DEBUG] generate_caption called with mode: {mode}")
        print(f"[DEBUG] image_data type: {type(image_data)}")
        print(f"[DEBUG] image_data content: {image_data}")
        
        if loaded_model is None:
            return "❌ Error: No model loaded. Please load a model first using the config file."
        
        # Handle different input formats from Gradio components
        image = None
        if image_data is None:
            return "❌ Error: No image data provided."
        
        # Check if it's a PIL Image directly
        if isinstance(image_data, Image.Image):
            print("[DEBUG] Received PIL Image directly")
            image = image_data
        # Check if it's a dict (from image_annotator component)
        elif isinstance(image_data, dict):
            print(f"[DEBUG] Received dict with keys: {list(image_data.keys())}")
            if 'image' in image_data:
                image_array = image_data['image']
                # Convert numpy array to PIL Image if needed
                if hasattr(image_array, 'shape') and len(image_array.shape) == 3:
                    print("[DEBUG] Converting numpy array to PIL Image")
                    image = Image.fromarray(image_array)
                else:
                    image = image_array
            elif 'background' in image_data:
                image_array = image_data['background']
                # Convert numpy array to PIL Image if needed
                if hasattr(image_array, 'shape') and len(image_array.shape) == 3:
                    print("[DEBUG] Converting numpy array to PIL Image")
                    image = Image.fromarray(image_array)
                else:
                    image = image_array
            else:
                return f"❌ Error: No image found in data. Available keys: {list(image_data.keys())}"
        # Check for tuple/list format (from ImageEditor component)
        elif isinstance(image_data, (tuple, list)) and len(image_data) >= 1:
            print(f"[DEBUG] Received tuple/list with {len(image_data)} elements")
            image = image_data[0]  # First element should be the image
            if not isinstance(image, Image.Image):
                # Sometimes the structure might be different, search for PIL Image
                for item in image_data:
                    if isinstance(item, Image.Image):
                        image = item
                        break
        else:
            return f"❌ Error: Unexpected data type: {type(image_data)}"
        
        if image is None:
            return "❌ Error: Image is None."
        
        # Convert PIL image if necessary
        if not isinstance(image, Image.Image):
            return "❌ Error: Invalid image format."
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if mode == "trace":
            return generate_trace_caption(image_data, image)
        elif mode == "bbox":
            return generate_bbox_caption(image_data, image)
        else:
            return f"❌ Error: Unknown mode: {mode}"
            
    except Exception as e:
        error_msg = f"❌ Error generating caption: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg


def generate_trace_caption(image_data, image) -> str:
    """Generate caption using traces."""
    global loaded_model
    
    try:
        # Process traces
        print("[DEBUG] Processing traces...")
        traces = process_image_trace_to_coordinates(image_data)
        print(f"[DEBUG] Found {len(traces)} traces")
        
        if not traces:
            # For debugging, let's generate a simple image caption instead of failing
            print("[DEBUG] No traces found, generating image caption instead")
            image_tensor = loaded_model.image_transforms(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = loaded_model(
                    image_tensor,
                    get_cls_capt=True,  # Get class caption as fallback
                    get_patch_capts=False,
                    get_avg_patch_capt=False
                )
            
            if 'cls_capt' in outputs:
                return f"🔍 No traces drawn. Image caption: {outputs['cls_capt']}"
            else:
                return "❌ Error: No traces detected. Please draw some traces on the image."
        
        print(f"Processing {len(traces)} traces")
        
        # Prepare image tensor
        image_tensor = loaded_model.image_transforms(image).unsqueeze(0).to(device)
        
        # Generate caption using the model
        with torch.no_grad():
            outputs = loaded_model(
                image_tensor,
                traces=traces,
                get_cls_capt=False,  # We want trace captions, not class captions
                get_patch_capts=False,
                get_avg_patch_capt=False
            )
        
        # Extract the trace captions
        if 'trace_capts' in outputs:
            captions = outputs['trace_capts']
            if isinstance(captions, list) and captions:
                captions = [cap.replace("<|startoftext|>", "").replace("<|endoftext|>", "") for cap in captions]
                # Join multiple captions if there are multiple traces
                if len(captions) == 1:
                    return f"Generated Caption: {captions[0]}"
                else:
                    formatted_captions = []
                    for i, caption in enumerate(captions, 1):
                        formatted_captions.append(f"Trace {i}: {caption}")
                    return "Generated Captions:\n" + "\n".join(formatted_captions)
            elif isinstance(captions, str):
                return f"Generated Caption: {captions}"
            else:
                return "❌ Error: No captions generated."
        else:
            return "❌ Error: Model did not return trace captions."
            
    except Exception as e:
        error_msg = f"❌ Error generating trace caption: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg


def generate_bbox_caption(image_data, image) -> str:
    """Generate caption using bounding boxes."""
    global loaded_model
    
    try:
        # Process bounding boxes
        print("[DEBUG] Processing bounding boxes...")
        bboxes = process_bounding_box_coordinates(image_data)
        print(f"[DEBUG] Found {len(bboxes)} bounding boxes")
        
        if not bboxes:
            # For debugging, let's generate a simple image caption instead of failing
            print("[DEBUG] No bounding boxes found, generating image caption instead")
            image_tensor = loaded_model.image_transforms(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = loaded_model(
                    image_tensor,
                    get_cls_capt=True,  # Get class caption as fallback
                    get_patch_capts=False,
                    get_avg_patch_capt=False
                )
            
            if 'cls_capt' in outputs:
                return f"🔍 No bounding boxes drawn. Image caption: {outputs['cls_capt']}"
            else:
                return "❌ Error: No bounding boxes detected. Please draw some bounding boxes on the image."
        
        print(f"Processing {len(bboxes)} bounding boxes")
        
        # Generate caption using the caption_bboxes method (as in eval_densecap.py)
        try:
            captions = loaded_model.caption_bboxes([image], [bboxes], crop_boxes=True)

            if isinstance(captions, list) and captions:
                if isinstance(captions[0], list):
                    captions = captions[0]  # Unwrap nested list if needed
                captions = [cap.replace("<|startoftext|>", "").replace("<|endoftext|>", "") for cap in captions]
                # Join multiple captions if there are multiple bboxes
                if len(captions) == 1:
                    return f"Generated Caption: {captions[0]}"
                else:
                    formatted_captions = []
                    for i, caption in enumerate(captions, 1):
                        formatted_captions.append(f"BBox {i}: {caption}")
                    return "Generated Captions:\n" + "\n".join(formatted_captions)
            elif isinstance(captions, str):
                return f"Generated Caption: {captions}"
            else:
                return "❌ Error: No captions generated."
                
        except Exception as e:
            print(f"Error using caption_bboxes method: {e}")
            # Fallback to regular forward method with bboxes
            image_tensor = loaded_model.image_transforms(image).unsqueeze(0).to(device)
            bbox_tensor = torch.tensor([bboxes]).to(device)
            
            with torch.no_grad():
                outputs = loaded_model(
                    image_tensor,
                    bboxes=bbox_tensor,
                    get_cls_capt=False,
                    get_patch_capts=False,
                    get_avg_patch_capt=False
                )
            
            if 'bbox_capts' in outputs:
                captions = outputs['bbox_capts']
                if isinstance(captions, list) and captions:
                    if isinstance(captions[0], list):
                        captions = captions[0]  # Unwrap nested list if needed
                    captions = [cap.replace("<|startoftext|>", "").replace("<|endoftext|>", "") for cap in captions]
                    if len(captions) == 1:
                        return f"Generated Caption: {captions[0]}"
                    else:
                        formatted_captions = []
                        for i, caption in enumerate(captions, 1):
                            formatted_captions.append(f"BBox {i}: {caption}")
                        return "Generated Captions:\n" + "\n".join(formatted_captions)
                elif isinstance(captions, str):
                    return f"Generated Caption: {captions}"
                else:
                    return "❌ Error: No captions generated."
            else:
                return "❌ Error: Model did not return bbox captions."
            
    except Exception as e:
        error_msg = f"❌ Error generating bbox caption: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return error_msg


def create_gradio_interface():
    """Create and configure the Gradio interface."""
    
    # Get example files
    example_images = get_example_images()
    example_configs = get_example_configs()


    custom_js = """
        <script>
        window.addEventListener("load", () => {
            // Hide Crop, Erase, and Color buttons
            const cropBtn   = document.querySelector('.image-editor__tool[title="Crop"]');
            const eraseBtn  = document.querySelector('.image-editor__tool[title="Erase"]');
            const colorBtn  = document.querySelector('.image-editor__tool[title="Color"]');

            [cropBtn, eraseBtn, colorBtn].forEach(btn => {
                console.log("Going to disable display for ", btn);
                if (btn) btn.style.display = "none";
            });

            // Optionally, select the Brush/Draft tool right away
            const brushBtn = document.querySelector('.image-editor__tool[title="Draw"]');
            console.log("Selecting brushbtn: ", brushBtn);
            if (brushBtn) brushBtn.click();
        });
        </script>
        """
    
    with gr.Blocks(
        title="Patchioner Trace Captioning Demo",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        """
    ) as demo:
        #gr.HTML(custom_js)  # inject custom JS
        
        gr.Markdown("""
        # 🎯 Patchioner Trace Captioning Demo
        
        This demo allows you to:
        1. **Select a captioning mode** (trace or bounding box)
        2. **Upload or select an image** from examples  
        3. **Draw traces or bounding boxes** on the image
        4. **Generate captions** describing the marked areas
        
        ## Instructions:
        1. Choose between Trace or BBox mode
        2. Upload an image or use one of the provided examples
        3. Use the appropriate tool to mark areas of interest in the image
        4. Click "Generate Caption" to get AI-generated descriptions
        
        **Model:** Using `mlp.karpathy.yaml` configuration (automatically loaded)
        """)
        
        # Initialize model status
        model_initialization_status = initialize_default_model()
        
        with gr.Row():
            gr.Markdown(f"**Model Status:** {model_initialization_status}")
        
        with gr.Row():
            mode_selector = gr.Radio(
                choices=["trace", "bbox"],
                value="trace",
                label="📋 Captioning Mode",
                info="Choose between trace-based or bounding box-based captioning",
                visible=False
            )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🖼️ Image Editor")
                
                # Image editor for drawing traces (default)
                image_editor = gr.ImageEditor(
                    label="Upload image and draw traces",
                    type="pil",
                    crop_size=None,
                    brush=gr.Brush(default_size=3, colors=["red", "blue", "green", "yellow", "purple"]),
                    visible=True,
                    #tools=["brush"],
                    height=600
                )
                
                # Image annotator for bounding boxes (hidden by default)
                image_annotator = foo_image_annotator( #gr.Image(
                    label="Upload image and draw bounding boxes",
                    visible=False,
                    #classes=["object"],
                    #type="bbox"
                    #tool="select"
                )
                
            with gr.Column():
                if example_images:
                    gr.Markdown("#### 📷 Or select from example images:")
                    example_gallery = gr.Gallery(
                        value=example_images,
                        label="Example Images",
                        show_label=True,
                        elem_id="gallery",
                        columns=3,
                        rows=2,
                        object_fit="contain",
                        height="auto"
                    )
        
        with gr.Row():
            generate_button = gr.Button("✨ Generate Caption", variant="primary", size="lg")
        
        with gr.Row():
            output_text = gr.Textbox(
                label="Generated Caption",
                placeholder="Generated caption will appear here...",
                lines=5,
                max_lines=10,
                interactive=False
            )
        
        # Event handlers
        def toggle_input_components(mode):
            """Toggle between image editor and annotator based on mode."""
            if mode == "trace":
                return gr.update(visible=True), gr.update(visible=False)
            else:  # bbox mode
                return gr.update(visible=False), gr.update(visible=True)
        
        def load_example_image_to_both(evt: gr.SelectData):
            """Load selected example image into both components."""
            try:
                example_images = get_example_images()
                if evt.index < len(example_images):
                    selected_image_path = example_images[evt.index]
                    img = Image.open(selected_image_path)
                    # For ImageEditor, return the PIL image directly
                    # For image_annotator, return dict format as expected by the component
                    annotated_format = {
                        "image": img,
                        "boxes": [],
                        "orientation": 0
                    }
                    return img, annotated_format
                return None, {"image": None, "boxes": [], "orientation": 0}
            except Exception as e:
                print(f"Error loading example image: {e}")
                return None, {"image": None, "boxes": [], "orientation": 0}
        
        def generate_caption_wrapper(mode, image_editor_data, image_annotator_data):
            """Wrapper to call generate_caption with the appropriate data based on mode."""
            if mode == "trace":
                return generate_caption(mode, image_editor_data)
            else:  # bbox mode
                return generate_caption(mode, image_annotator_data)
        
        # Connect event handlers
        mode_selector.change(
            fn=toggle_input_components,
            inputs=mode_selector,
            outputs=[image_editor, image_annotator]
        )
        
        generate_button.click(
            fn=generate_caption_wrapper,
            inputs=[mode_selector, image_editor, image_annotator],
            outputs=output_text
        )
        
        if example_images:
            example_gallery.select(
                fn=load_example_image_to_both,
                outputs=[image_editor, image_annotator]
            )
        
        gr.Markdown("""
        ### 💡 Tips:
        - **Mode Selection**: Switch between trace and bounding box modes based on your needs
        - **Trace Mode**: Draw continuous lines over areas you want to describe
        - **BBox Mode**: Draw rectangular bounding boxes around objects of interest
        - **Multiple Areas**: Create multiple traces/boxes for different objects to get individual captions
        - **Model Performance**: First load may take some time as weights are downloaded
        
        ### 🔧 Technical Details:
        - **Trace Mode**: Converts drawings to normalized (x, y) coordinates with timestamps
        - **BBox Mode**: Uses bounding box coordinates for region-specific captioning
        - **Model Architecture**: Uses `mlp.karpathy.yaml` configuration with CLIP and ViT components
        - **Processing**: Each trace/bbox is processed separately to generate corresponding captions
        """)
    
    return demo


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Patchioner Trace Captioning Demo")
    parser.add_argument("--port", type=int, default=4141, help="Port to run the Gradio app on")
    args = parser.parse_args()

    print("Starting Patchioner Trace Captioning Demo...")
    print(f"Using device: {device}")
    print(f"Default model: {DEFAULT_MODEL_CONFIG}")
    print(f"Example images directory: {EXAMPLE_IMAGES_DIR}")
    print(f"Configs directory: {CONFIGS_DIR}")
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=True,
        debug=True
    )
