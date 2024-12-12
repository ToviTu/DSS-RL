import torch
from torch import nn
from torchvision.models import resnet18
import torchvision.transforms as transforms

class BaseFeature:
    def __init__(self):
        pass

    def __call__(self, x):
        return x

class FlatteningFeature(BaseFeature):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        if x.ndim == 4:
            bcz = x.size(0)
            return x.view(bcz, -1)
        else:
            return x.view(-1)

class ResNetFeature(BaseFeature):
    def __init__(self, weights='DEFAULT', device=None, layer=-1):
        super().__init__()
        # Choose device: 'cuda' if available, else 'cpu'
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load a pretrained ResNet model and remove the last fully connected layer
        self.model = resnet18(weights=weights)
        self.model = nn.Sequential(*list(self.model.children())[:layer])  # Remove the final layer
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Define the image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet expects 224x224 input
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image):
        # Preprocess the image and encode it as a feature vector
        image = self.transform(image).to(self.device)
        with torch.no_grad():
            features = self.model(image)
        return features.squeeze((2, 3)).cpu()  # Flatten and convert to numpy array

class SentenceTransformerFeature(BaseFeature):
    def __init__(self, model_name='all-mpnet-base-v2', device=None):
        super().__init__()
        # Choose device: 'cuda' if available, else 'cpu'
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load a pretrained SentenceTransformer model
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

    def __call__(self, sentence):
        # Encode the sentence as a feature vector
        if isinstance(sentence, str):
            sentence = [sentence]

        with torch.no_grad():
            features = self.model.encode(sentence, convert_to_tensor=True)
        return  features.squeeze().cpu() # Convert to numpy array