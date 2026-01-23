# TFDAT
[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/J3J3BCC3L)

TFDAT (Temporal FDAT) is a custom VSR architecture. This is a continuation of my previous VSR architectures, namely TSPANv2. TFDAT is a major step up in every way, providing significantly better quality, temporal coherency, and even inference speed! 

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

You can use TFDAT through [Vapourkit](https://github.com/Kim2091/vapourkit) (preferred), the included GUI (directions below), or the command line.

<img width="2033" height="1248" alt="image" src="https://github.com/user-attachments/assets/821bf399-cf6c-4170-a492-1df5ebef305f" />


## TensorRT

For high-performance inference, refer to the [TensorRT guide](tensorrt/README.md).

### GUI Usage

For an easy-to-use experience with PyTorch or ONNX models, launch the GUI:

```bash
python vsr_gui.py
```

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
Unlike my previous repositories, ONNX conversion is now done within [traiNNer-redux](https://github.com/Kim2091/traiNNer-redux-1) instead. Follow these directions: https://trainner-redux.readthedocs.io/en/latest/getting_started.html#convert-models-to-onnx

## Credits (thanks all!)
Thank you to leobby and Bendel for testing the arch!
- Folder structure and video processing code is derived from [SCUNet](https://github.com/aaf6aa/SCUNet)

- The TFDAT architecture is based on [FDAT](https://github.com/stinkybread/FDAT) with extensive modifications



