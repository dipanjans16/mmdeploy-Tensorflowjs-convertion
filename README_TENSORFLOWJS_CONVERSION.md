# TensorFlow.js Pose Detection Pipeline

This repository contains a complete conversion pipeline that transforms the RTMPose model from ONNX format to TensorFlow.js, enabling real-time human pose detection in web browsers.

## 🌟 Features

- **Complete Pipeline Conversion**: Converts preprocessing, inference, and postprocessing to TensorFlow operations
- **TensorFlow.js Ready**: Exports models that run directly in web browsers
- **Zero Dependencies**: No server required - everything runs client-side
- **Multiple Model Types**: Support for both complete and inference-only models
- **Web Demo Included**: Ready-to-use HTML demo with pose visualization

## 📁 Files Overview

### Core Conversion Scripts
- `pytorch_free_inference_tf_with_tf_preprocessing.py` - Main TensorFlow.js converter
- `tensorflow_js_complete_pipeline.py` - Advanced complete pipeline converter
- `run_tensorflowjs_conversion.py` - Simple conversion script
- `demo_tensorflowjs_conversion.py` - Full demo with web interface

### Original Pipeline
- `pytorch_free_inference_tf.py` - TensorFlow inference with OpenCV preprocessing
- `tensorflow_complete_pipeline.py` - Complete TensorFlow pipeline
- `complete_numpy_postprocess.py` - NumPy-based postprocessing

## 🚀 Quick Start

### Step 1: Convert ONNX to TensorFlow.js

```bash
# Simple conversion (recommended)
python run_tensorflowjs_conversion.py

# Or with custom paths
python pytorch_free_inference_tf_with_tf_preprocessing.py \
    --tf-model temp_comparison_tf_model \
    --output tensorflowjs_pose_model \
    --model-type inference_only
```

### Step 2: Test the Web Demo

```bash
# Start a web server
python -m http.server 8000

# Open in browser
http://localhost:8000/tensorflowjs_pose_model/demo.html
```

## 📋 Prerequisites

### Python Dependencies
```bash
pip install numpy==1.24.5
pip install opencv-python matplotlib mediapipe opencv-python
pip install tensorflow>=2.8.0
pip install tensorflowjs>=3.18.0
pip install opencv-python
pip install numpy
```

### For ONNX Conversion (if needed)
```bash
pip install onnx>=1.12.0
pip install onnx2tf
```

## 🔧 Model Types

### 1. Inference-Only Model (Recommended)
- **Size**: Smaller (~50-100MB)
- **Speed**: Faster loading
- **Preprocessing**: Done in JavaScript
- **Postprocessing**: Done in JavaScript
- **Use Case**: Most web applications

```python
pipeline.convert_to_tensorflowjs(
    tf_model_path,
    output_path,
    model_type='inference_only',
    quantization='float16'
)
```

### 2. Complete Model (Experimental)
- **Size**: Larger
- **Speed**: Slower loading, but potentially faster inference
- **Preprocessing**: Embedded in model
- **Postprocessing**: Embedded in model
- **Use Case**: When you want everything in the model

```python
pipeline.convert_to_tensorflowjs(
    tf_model_path,
    output_path,
    model_type='complete',
    quantization='float16'
)
```

## 🌐 JavaScript Usage

### Basic Usage with Helper Class

```html
<!DOCTYPE html>
<html>
<head>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.0.0/dist/tf.min.js"></script>
    <script src="rtmpose_helper.js"></script>
</head>
<body>
    <script>
        async function runPoseDetection() {
            // Load model
            const model = await tf.loadGraphModel('./model.json');
            
            // Create helper
            const poseHelper = new RTMPoseHelper();
            
            // Get image element
            const imageElement = document.getElementById('myImage');
            
            // Run pose detection
            const result = await poseHelper.predict(model, imageElement);
            
            console.log('Keypoints:', result.keypoints);
            console.log('Scores:', result.scores);
        }
    </script>
</body>
</html>
```

### Manual Usage (Without Helper)

```javascript
// Load model
const model = await tf.loadGraphModel('./model.json');

// Preprocess image
const preprocessed = tf.tidy(() => {
    let tensor = tf.browser.fromPixels(imageElement);
    tensor = tf.image.resizeBilinear(tensor.expandDims(0), [256, 192]);
    tensor = tensor.sub([123.675, 116.28, 103.53]).div([58.395, 57.12, 57.375]);
    return tensor.transpose([0, 3, 1, 2]); // NHWC to NCHW
});

// Run inference
const [simccX, simccY] = await model.executeAsync(preprocessed);

// Postprocess (find keypoints)
const xLocs = simccX.argMax(-1);
const yLocs = simccY.argMax(-1);
const keypoints = tf.stack([xLocs, yLocs], -1).div(2.0); // Scale by simcc_split_ratio

const scores = tf.minimum(simccX.max(-1), simccY.max(-1));

// Convert to JavaScript arrays
const keypointsArray = await keypoints.array();
const scoresArray = await scores.array();

// Cleanup tensors
preprocessed.dispose();
simccX.dispose();
simccY.dispose();
keypoints.dispose();
scores.dispose();
```

## 📊 Model Specifications

### Input
- **Format**: RGB image
- **Size**: Variable (automatically resized to 256x192)
- **Type**: uint8 or float32
- **Channels**: 3 (RGB)

### Output
- **Keypoints**: Array of [x, y] coordinates
- **Scores**: Confidence scores for each keypoint
- **Count**: 133 keypoints (COCO-WholeBody format)
  - 17 body keypoints
  - 68 face keypoints
  - 42 hand keypoints (21 per hand)
  - 6 foot keypoints

### Performance
- **Browser**: Chrome, Firefox, Safari (latest versions)
- **Speed**: ~100-500ms per frame (depends on hardware)
- **Memory**: ~200-500MB (including TensorFlow.js)

## 🎯 Keypoint Format

The model outputs 133 keypoints in COCO-WholeBody format:

```javascript
// Body keypoints (0-16)
const bodyKeypoints = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
];

// Face keypoints (17-84) - 68 facial landmarks
// Hand keypoints (85-126) - 21 per hand
// Foot keypoints (127-132) - 6 total
```

## 🔧 Advanced Configuration

### Quantization Options
```python
# Float16 (recommended) - smaller size, slight precision loss
quantization='float16'

# Int8 - smallest size, more precision loss
quantization='int8'

# None - full precision, largest size
quantization='none'
```

### Custom Input Size
```python
pipeline = CompleteTensorFlowJSPipeline(input_size=(256, 192))  # width, height
```

### Custom Preprocessing Parameters
```javascript
const helper = new RTMPoseHelper();
helper.inputSize = [256, 192];
helper.mean = [123.675, 116.28, 103.53];
helper.std = [58.395, 57.12, 57.375];
helper.simccSplitRatio = 2.0;
```

## 🐛 Troubleshooting

### Common Issues

1. **Model fails to load**
   ```
   Error: Failed to fetch model.json
   ```
   - Solution: Ensure you're serving files from a web server, not opening HTML directly
   - Start server: `python -m http.server 8000`

2. **CORS errors**
   ```
   Access to fetch at 'model.json' blocked by CORS policy
   ```
   - Solution: Serve files from the same domain or configure CORS headers

3. **Out of memory errors**
   ```
   Error: Cannot allocate tensor
   ```
   - Solution: Use quantized models (`float16` or `int8`)
   - Close other browser tabs
   - Use smaller input images

4. **Slow performance**
   - Use GPU acceleration: Ensure TensorFlow.js can access WebGL
   - Check browser console for WebGL warnings
   - Consider using smaller quantized models

### Browser Compatibility
- ✅ Chrome 70+
- ✅ Firefox 65+
- ✅ Safari 12+
- ✅ Edge 79+

### Performance Tips
1. **Preload the model** during page initialization
2. **Reuse tensors** when possible
3. **Use tf.tidy()** to prevent memory leaks
4. **Dispose of tensors** after use
5. **Process frames** at 30fps or less for real-time applications

## 📂 Output Structure

After conversion, you'll get:
```
tensorflowjs_pose_model/
├── inference_only/
│   ├── model.json              # Model architecture
│   ├── group1-shard1of2.bin    # Model weights (part 1)
│   ├── group1-shard2of2.bin    # Model weights (part 2)
│   ├── metadata.json           # Model information
│   └── rtmpose_helper.js       # JavaScript helper functions
├── complete/                   # (if successful)
│   ├── model.json
│   ├── *.bin
│   └── metadata.json
└── demo.html                   # Web demo page
```

## 📖 Further Reading

- [TensorFlow.js Documentation](https://www.tensorflow.org/js)
- [RTMPose Paper](https://arxiv.org/abs/2303.07399)
- [COCO-WholeBody Dataset](https://github.com/jin-s13/COCO-WholeBody)
- [MMPose Documentation](https://mmpose.readthedocs.io/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the demo
5. Submit a pull request

## 📄 License

This project follows the same license as the original MMPose and MMDeploy projects.

---

**Happy pose detecting! 🕺💃** 