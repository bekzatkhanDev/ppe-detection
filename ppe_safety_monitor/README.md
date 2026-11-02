# PPE Safety Monitor

Real-time PPE (Personal Protective Equipment) safety monitoring desktop app
using a pre-trained YOLO model, with two interchangeable weights modes:

| Mode | Weights file        | Backend                          | Notes                              |
|------|---------------------|----------------------------------|------------------------------------|
| `pt` | `models/best.pt`    | PyTorch (ultralytics + torch)    | Default; full-featured             |
| `onnx` | `models/best.onnx` | ONNX Runtime                     | Lighter/faster on CPU, no torch    |

## Setup

```bash
pip install -r requirements.txt
```

Place your trained model files at `models/best.pt` and/or `models/best.onnx`.
To export an ONNX version from your PyTorch weights:

```bash
yolo export model=models/best.pt format=onnx
# or in Python:
# from ultralytics import YOLO; YOLO("models/best.pt").export(format="onnx")
```

## Run

```bash
python main.py               # uses DEFAULT_MODEL_MODE from src/config.py ("pt")
python main.py --mode pt     # PyTorch weights (models/best.pt)
python main.py --mode onnx   # ONNX weights  (models/best.onnx)
```

You can also switch the mode live from the "Model weights" dropdown in the
Controls panel (monitoring restarts automatically when changed while running).

The active mode is resolved in `src/config.py` (`MODEL_MODES`,
`DEFAULT_MODEL_MODE`); set `DEVICE` there to force `"cpu"` / `"cuda:0"`,
or leave it `None` for automatic selection.
