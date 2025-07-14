#!/usr/bin/env python3
"""
Integration script to convert your existing pipeline to a complete TensorFlow.js model.
This script modifies your existing code to create a single end-to-end model.

uv pip install tensorflow==2.15.0  
$env:UV_HTTP_TIMEOUT = "60"        
uv pip install tensorflowjs     
pip install tensorflow==2.15.0 tensorflowjs==4.17.0
pip install tensorflow_decision_forests==1.8.1
pip install opencv-python==4.8.1.78     
pip install numpy==1.25.0  


pip install tensorflow==2.13.0
pip install tensorflowjs
pip install tensorflow-decision-forests==1.5.0
"""

import tensorflow as tf
import tensorflowjs
import numpy as np
import cv2
import os
from typing import Tuple, Dict, Any
import json
import argparse

class EnhancedTensorFlowPosePipeline:
    """Enhanced version that integrates with your existing code."""
    
    def __init__(self, input_size: Tuple[int, int] = (192, 256), simcc_split_ratio: float = 2.0):
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio
        self.model = None
        
    def create_tf_preprocessing_ops(self):
        """Create TensorFlow preprocessing operations that match your original preprocess function."""
        
        @tf.function
        def tf_preprocess(image):
            """TensorFlow version of your preprocess function."""
            
            # Input: [batch, height, width, channels] uint8
            # Output: [batch, channels, height, width] float32 + transformation info
            
            batch_size = tf.shape(image)[0]
            height = tf.cast(tf.shape(image)[1], tf.float32)
            width = tf.cast(tf.shape(image)[2], tf.float32)
            
            # Calculate center and scale (matching your original logic)
            center_x = width / 2.0
            center_y = height / 2.0
            scale_x = width
            scale_y = height
            
            # Aspect ratio adjustment
            w_input = tf.constant(self.input_size[0], dtype=tf.float32)
            h_input = tf.constant(self.input_size[1], dtype=tf.float32)
            aspect_ratio = w_input / h_input
            
            # Apply aspect ratio correction
            scale_x = tf.where(width / height > aspect_ratio, width, height * aspect_ratio)
            scale_y = tf.where(width / height > aspect_ratio, width / aspect_ratio, height)
            
            # Add padding (1.25 factor from your original code)
            scale_x = scale_x * 1.25
            scale_y = scale_y * 1.25
            
            # Resize image to input size
            image_f32 = tf.cast(image, tf.float32)
            resized_image = tf.image.resize(image_f32, [self.input_size[1], self.input_size[0]])
            
            # Normalize using your exact values
            mean = tf.constant([123.675, 116.28, 103.53], dtype=tf.float32)
            std = tf.constant([58.395, 57.12, 57.375], dtype=tf.float32)
            normalized_image = (resized_image - mean) / std
            
            # Convert from HWC to CHW format
            normalized_image = tf.transpose(normalized_image, [0, 3, 1, 2])
            
            return normalized_image, center_x, center_y, scale_x, scale_y
        
        return tf_preprocess
    
    def create_tf_postprocessing_ops(self):
        """Create TensorFlow postprocessing operations that match CompleteNumPyPoseProcessor."""
        
        @tf.function
        def tf_postprocess(simcc_x, simcc_y, center_x, center_y, scale_x, scale_y):
            """TensorFlow version of CompleteNumPyPoseProcessor.process_onnx_outputs."""
            
            # Get dimensions
            batch_size = tf.shape(simcc_x)[0]
            num_keypoints = tf.shape(simcc_x)[1]
            simcc_x_size = tf.cast(tf.shape(simcc_x)[2], tf.float32)
            simcc_y_size = tf.cast(tf.shape(simcc_y)[2], tf.float32)
            
            # Find maximum confidence positions
            x_locs = tf.argmax(simcc_x, axis=2)
            y_locs = tf.argmax(simcc_y, axis=2)
            
            # Get confidence scores
            x_confs = tf.reduce_max(simcc_x, axis=2)
            y_confs = tf.reduce_max(simcc_y, axis=2)
            
            # Convert to float32 for calculations
            x_locs_f = tf.cast(x_locs, tf.float32)
            y_locs_f = tf.cast(y_locs, tf.float32)
            
            # Calculate scale factors for coordinate mapping
            w_input = tf.constant(self.input_size[0], dtype=tf.float32)
            h_input = tf.constant(self.input_size[1], dtype=tf.float32)
            
            # Map from SimCC coordinates to input image coordinates
            x_coords = (x_locs_f / (simcc_x_size - 1)) * w_input
            y_coords = (y_locs_f / (simcc_y_size - 1)) * h_input
            
            # Transform back to original image coordinates
            # This implements the inverse of the affine transformation
            x_coords_orig = (x_coords - w_input / 2.0) * (scale_x / w_input) + center_x
            y_coords_orig = (y_coords - h_input / 2.0) * (scale_y / h_input) + center_y
            
            # Stack coordinates
            keypoints = tf.stack([x_coords_orig, y_coords_orig], axis=2)
            
            # Calculate final confidence scores (minimum of x and y confidences)
            confidence_scores = tf.minimum(x_confs, y_confs)
            
            return keypoints, confidence_scores
        
        return tf_postprocess
    
    def create_complete_model_from_existing(self, tf_model_path: str):
        """Create complete model using your existing TensorFlow model."""
        
        # Load the existing TensorFlow model
        print(f"Loading TensorFlow model from: {tf_model_path}")
        base_model = tf.saved_model.load(tf_model_path)
        
        # Debug: Print model signature information
        print("Model signature information:")
        if hasattr(base_model, 'signatures'):
            for sig_name, sig in base_model.signatures.items():
                print(f"  Signature '{sig_name}':")
                if hasattr(sig, 'structured_input_signature'):
                    input_spec = sig.structured_input_signature[1]
                    print(f"    Input spec: {input_spec}")
                if hasattr(sig, 'structured_outputs'):
                    output_spec = sig.structured_outputs
                    print(f"    Output spec: {output_spec}")
        else:
            print("  No signatures found")
        
        # Get preprocessing and postprocessing functions
        preprocess_fn = self.create_tf_preprocessing_ops()
        postprocess_fn = self.create_tf_postprocessing_ops()
        
        @tf.function
        def complete_inference(image):
            # Preprocessing
            preprocessed_image, center_x, center_y, scale_x, scale_y = preprocess_fn(image)
            
            # Get the input name from the model signature
            # This is how the existing TensorFlow pipeline calls the model
            try:
                # Try to get input name from signature
                if hasattr(base_model, 'signatures') and 'serving_default' in base_model.signatures:
                    signature = base_model.signatures['serving_default']
                    input_name = list(signature.structured_input_signature[1].keys())[0]
                    outputs = signature(**{input_name: preprocessed_image})
                else:
                    # Fallback: try common input names
                    for input_name in ['input', 'x', 'input_1', 'serving_default_input_1']:
                        try:
                            outputs = base_model(**{input_name: preprocessed_image})
                            break
                        except:
                            continue
                    else:
                        # Last resort: try positional argument
                        outputs = base_model(preprocessed_image)
            except Exception as e:
                # If all fails, try to inspect the model structure
                raise ValueError(f"Could not call model with any input format. Error: {e}")
            
            # Extract SimCC outputs (adjust based on your model structure)
            if isinstance(outputs, dict):
                simcc_x = outputs.get('simcc_x', list(outputs.values())[0])
                simcc_y = outputs.get('simcc_y', list(outputs.values())[1])
            elif isinstance(outputs, (list, tuple)):
                simcc_x = outputs[0]
                simcc_y = outputs[1]
            else:
                # Handle single output case
                raise ValueError("Unexpected output format from base model")
            
            # Postprocessing
            keypoints, confidence_scores = postprocess_fn(
                simcc_x, simcc_y, center_x, center_y, scale_x, scale_y
            )
            
            return {
                'keypoints': keypoints,
                'confidence_scores': confidence_scores
            }
        
        return complete_inference
    
    def save_complete_model(self, tf_model_path: str, output_path: str):
        """Save the complete end-to-end model."""
        
        complete_model_fn = self.create_complete_model_from_existing(tf_model_path)
        
        # Define input signature for uint8 images
        input_signature = tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.uint8)
        
        # Create concrete function
        concrete_function = complete_model_fn.get_concrete_function(input_signature)
        
        # Save the model
        tf.saved_model.save(
            concrete_function,
            output_path,
            signatures={'serving_default': concrete_function}
        )
        
        print(f"Complete model saved to: {output_path}")
        return output_path
    
    def convert_to_tensorflowjs(self, saved_model_path: str, output_path: str):
        """Convert to TensorFlow.js with optimizations."""
        
        # Convert with optimizations
        tfjs.converters.convert_tf_saved_model(
            saved_model_path,
            output_path,
            signature_name='serving_default',
            saved_model_tags=['serve'],
            quantization_bytes=2,  # Quantize to reduce model size
            strip_debug_ops=True,
            control_flow_v2=True
        )
        
        print(f"TensorFlow.js model saved to: {output_path}")
        
        # Create model info file
        self.create_model_info_file(output_path)
        
        return output_path
    def create_model_info_file(self, model_path: str):
        """Create a JSON file with model information."""
        
        model_info = {
            "model_type": "pose_detection",
            "input_size": self.input_size,
            "simcc_split_ratio": self.simcc_split_ratio,
            "input_format": "uint8_image",
            "output_format": {
                "keypoints": "float32[batch, num_keypoints, 2]",
                "confidence_scores": "float32[batch, num_keypoints]"
            },
            "usage": {
                "input": "RGB image as uint8 tensor",
                "preprocessing": "Built-in (automatic)",
                "postprocessing": "Built-in (automatic)"
            }
        }
        
        info_path = os.path.join(model_path, 'model_info.json')
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"Model info saved to: {info_path}")
    
    def test_complete_model(self, model_path: str, test_image_path: str):
        """Test the complete model with a sample image."""
        
        # Load the complete model
        model = tf.saved_model.load(model_path)
        
        # Load and preprocess test image
        image = cv2.imread(test_image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor
        image_tensor = tf.convert_to_tensor(image_rgb)
        image_batch = tf.expand_dims(image_tensor, 0)
        
        # Run inference
        results = model(image_batch)
        
        # Extract results
        keypoints = results['keypoints'].numpy()
        confidence_scores = results['confidence_scores'].numpy()
        
        print(f"Detected {keypoints.shape[1]} keypoints")
        print(f"Confidence scores shape: {confidence_scores.shape}")
        print(f"Average confidence: {np.mean(confidence_scores):.4f}")


def main():
    """Main function to run the TensorFlow.js conversion pipeline."""
    parser = argparse.ArgumentParser(description='Convert pose detection model to TensorFlow.js')
    parser.add_argument('--tf_model_path', type=str, default='temp_comparison_tf_model',
                        help='Path to the TensorFlow SavedModel')
    parser.add_argument('--output_path', type=str, default='tfjs_complete_model',
                        help='Output path for the complete TensorFlow model')
    parser.add_argument('--tfjs_output_path', type=str, default='tfjs_web_model',
                        help='Output path for the TensorFlow.js model')
    parser.add_argument('--test_image', type=str, default='output_pytorch_free.jpg',
                        help='Path to test image')
    parser.add_argument('--input_width', type=int, default=192,
                        help='Model input width')
    parser.add_argument('--input_height', type=int, default=256,
                        help='Model input height')
    parser.add_argument('--skip_conversion', action='store_true',
                        help='Skip TensorFlow.js conversion, only test existing model')
    
    args = parser.parse_args()
    
    # Initialize the pipeline
    pipeline = EnhancedTensorFlowPosePipeline(
        input_size=(args.input_width, args.input_height)
    )
    
    try:
        if not args.skip_conversion:
            print("Step 1: Creating complete end-to-end model...")
            
            # Check if TensorFlow model exists
            if not os.path.exists(args.tf_model_path):
                print(f"Error: TensorFlow model not found at {args.tf_model_path}")
                print("Available models:")
                if os.path.exists('tensorflow_models'):
                    for item in os.listdir('tensorflow_models'):
                        if os.path.isdir(os.path.join('tensorflow_models', item)):
                            print(f"  - tensorflow_models/{item}")
                return
            
            # Save complete model
            complete_model_path = pipeline.save_complete_model(
                args.tf_model_path, 
                args.output_path
            )
            
            print("\nStep 2: Converting to TensorFlow.js...")
            
            # Convert to TensorFlow.js
            tfjs_path = pipeline.convert_to_tensorflowjs(
                complete_model_path,
                args.tfjs_output_path
            )
            
            if tfjs_path:
                print(f"\n✅ Complete conversion successful!")
                print(f"TensorFlow.js model saved to: {tfjs_path}")
            else:
                print("\n⚠️  TensorFlow.js conversion failed, but complete TensorFlow model was created successfully")
                print(f"📁 Complete TensorFlow model available at: {complete_model_path}")
        
        # Test the model
        if args.test_image and os.path.exists(args.test_image):
            print(f"\nStep 3: Testing model with {args.test_image}...")
            
            test_model_path = args.output_path if not args.skip_conversion else args.tf_model_path
            if os.path.exists(test_model_path):
                pipeline.test_complete_model(test_model_path, args.test_image)
            else:
                print(f"Model not found at {test_model_path}")
        else:
            if args.test_image:
                print(f"Test image not found: {args.test_image}")
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
