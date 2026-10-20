AI-05 MODEL HANDOFF

Scope: educational two-class Waterbirds classifier; 0=landbird, 1=waterbird.

From within one model folder:
    from predict import Predictor
    predictor = Predictor('model.pt', device='cpu')
    result = predictor.predict('/path/to/bird.jpg')
    comparison = predictor.compare('/path/to/original.png', '/path/to/edited.png')

Load once per application process, then reuse. Use the supplied predictor and config so RGB conversion,
EXIF orientation, full-frame resize, normalization, labels and model.eval() are consistent.
requirements.txt records core versions. Install a matching torch/torchvision pair for your OS/device;
for CUDA-specific installation use https://pytorch.org/get-started/locally/.
Only load checkpoints you trust; this adapter uses weights_only=True.

Scores are softmax outputs, not calibrated certainty. A score drop is tracked for the same class before
and after an edit. Group failure gaps and controlled edit sensitivity support limited claims, not a
universal causal or safety verdict. Report edit quality, controls, sample counts and uncertainty.

Dataset: https://github.com/kohpangwei/group_DRO#waterbirds
Label/schema reference: https://github.com/p-lambda/wilds/blob/main/wilds/datasets/waterbirds_dataset.py
Training: https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
Dataset source terms restrict use to non-commercial research and education (see WILDS loader).
