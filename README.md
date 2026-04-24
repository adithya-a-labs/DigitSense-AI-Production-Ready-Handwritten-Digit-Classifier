# DigitSense AI

DigitSense AI is a lean, GPU-aware MNIST classifier built with PyTorch and Streamlit. The repository includes a compact CNN, a training pipeline, an evaluation script, and a small app for interactive inference.

## Structure

- `src/model.py`: CNN architecture
- `src/train.py`: training loop, checkpointing, and metric plots
- `src/evaluate.py`: test evaluation and confusion matrix export
- `src/predict.py`: cached model loading and inference helper
- `src/utils.py`: shared paths, transforms, device handling, and image preprocessing
- `app/app.py`: Streamlit UI

## Configuration

Training defaults live in `config.yaml`:

- `epochs`
- `batch_size`
- `learning_rate`
- `num_workers`
- `dropout`
- `seed`

## Setup

```bash
python -m venv venv
```

Install the matching PyTorch build for your machine, then install the project dependencies:

```bash
pip install -r requirements.txt
```

The code uses CUDA automatically when the installed PyTorch build can execute on the GPU, and falls back to CPU otherwise.

## Run

Train the model:

```bash
python src/train.py
```

Evaluate the trained checkpoint:

```bash
python src/evaluate.py
```

Launch the app:

```bash
streamlit run app/app.py
```

## Outputs

- `outputs/model.pth`
- `outputs/plots/loss.png`
- `outputs/plots/accuracy.png`
- `outputs/confusion_matrix.png`

MNIST is cached locally under `.data/mnist/`.
