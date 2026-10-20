import torch
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from PIL import Image, ImageOps

class Predictor:
    def __init__(self, checkpoint_path, device='cpu'):
        self.device = torch.device(device)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['architecture'] != 'resnet18':
            raise ValueError('This adapter supports resnet18 checkpoints only')
        self.classes = checkpoint['class_names']
        self.run_name = checkpoint['run_name']
        cfg = checkpoint['preprocessing']
        if cfg['resize'] != 'full_frame_square' or cfg['interpolation'] != 'bilinear':
            raise ValueError('Unsupported preprocessing')
        self.transform = transforms.Compose([
            transforms.Resize((cfg['size'], cfg['size']),
                              interpolation=InterpolationMode.BILINEAR, antialias=True),
            transforms.ToTensor(), transforms.Normalize(cfg['mean'], cfg['std'])
        ])
        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, image_path):
        with Image.open(image_path) as image:
            tensor = self.transform(ImageOps.exif_transpose(image).convert('RGB'))
        probabilities = self.model(tensor.unsqueeze(0).to(self.device)).softmax(1)[0].cpu().tolist()
        predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {'model_version': self.run_name, 'predicted_label': predicted,
                'predicted_class': self.classes[predicted],
                'probabilities': {name: float(probabilities[i]) for i, name in enumerate(self.classes)}}

    def compare(self, original_path, edited_path):
        original, edited = self.predict(original_path), self.predict(edited_path)
        fixed_class = original['predicted_class']
        return {'model_version': self.run_name, 'original': original, 'edited': edited,
                'label_flipped': original['predicted_label'] != edited['predicted_label'],
                'tracked_class': fixed_class,
                'original_class_score_drop': original['probabilities'][fixed_class]
                                             - edited['probabilities'][fixed_class]}
