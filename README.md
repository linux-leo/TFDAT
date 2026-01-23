# TFDAT
TFDAT (Temporal FDAT) is a custom VSR architecture. This is a continuation of my previous VSR architectures, namely TFDAT. TFDAT is a major step up in every way, providing significantly better quality, temporal coherency, and even inference speed! 

This arch has support for PyTorch, ONNX, and TensorRT.

This repository hosts the Pytorch inference code. To train a TFDAT model, you'll want to use [traiNNer-redux](https://github.com/Kim2091/traiNNer-redux-1) with the TFDAT config and a video dataset. For easier inference than the GUI provided in this codebase, try out [Vapourkit](https://github.com/Kim2091/vapourkit). To make a video dataset, try my other tool, [video destroyer](https://github.com/Kim2091/video-destroyer).

## Getting Started

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/Kim2091/TFDAT
    ```

2.  **Install PyTorch with CUDA**:
    Follow the instructions at [pytorch.org](https://pytorch.org/get-started/locally/).

3.  **Install required packages**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

You can use TFDAT through the included graphical user interface (GUI), [Vapourkit](https://github.com/Kim2091/vapourkit) (preferred), or the command line.

## TensorRT

For high-performance inference, refer to the [TensorRT guide](tensorrt/README.md).

### GUI Usage

For an easy-to-use experience with PyTorch or ONNX models, launch the GUI:

```bash
python vsr_gui.py
```

<img width="602" height="698" alt="image" src="https://github.com/user-attachments/assets/744fd695-3fe8-4dc7-b52c-f3bca423e13c" />


### Command-Line Usage

For more advanced control, you can use the command-line scripts.

**Video upscaling (PyTorch)**:
```bash
python test_vsr.py --model_path pretrained_models/tfdat.pth --input path/to/video.mp4 --output path/to/output.mp4
```

Key arguments for `test_vsr.py` and `test_onnx.py`:
-   `--video_codec`: Specify the video codec (e.g., `libx264`, `libx265`).
-   `--crf`: Set the Constant Rate Factor for quality (for `libx264`/`libx265`).

## ONNX Conversion
Unlike my previous repositories, ONNX conversion is now done within traiNNer-redux instead. Follow these directions: https://trainner-redux.readthedocs.io/en/latest/getting_started.html#convert-models-to-onnx

## Credits (thanks all!)
Thank you to leobby and Bendel for testing the arch!
- Folder structure and video processing code is derived from [SCUNet](https://github.com/aaf6aa/SCUNet)
- Uses a modified version of [FDAT](https://github.com/stinkybread/FDAT)