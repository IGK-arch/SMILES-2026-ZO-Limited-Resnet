import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

def init_last_layer(layer: nn.Linear) -> None:
    """
    Weight Imprinting: Initialization of head weights using average features (centroids).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. We take a pure ResNet18 to extract the features
    backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    backbone.fc = nn.Identity()
    backbone.to(device)
    backbone.eval()

    # 2. We load the training data (without heavy augmentation for the purity of the signs)
    transform = T.Compose([T.Resize(224), T.ToTensor(), 
                           T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    features_sum = torch.zeros(100, 512).to(device)
    counts = torch.zeros(100).to(device)

    print("Computing centroids for Imprinting...")
    with torch.no_grad():
        # We calculate the sum of features for each class (we use the first 5000 images for speed)
        for i, (inputs, targets) in enumerate(loader):
            if i > 40: break # This is enough for excellent initialization.
            feats = backbone(inputs.to(device))
            for f, t in zip(feats, targets):
                features_sum[t] += f
                counts[t] += 1

    # 3. We calculate the average and normalize
    for i in range(100):
        if counts[i] > 0:
            features_sum[i] /= counts[i]
    
    # Normalization of weights (Cosine Similarity works better in transfer learning)
    features_sum = torch.nn.functional.normalize(features_sum, dim=1)
    
    # 4. Copy it to the layer
    layer.weight.data.copy_(features_sum)
    nn.init.zeros_(layer.bias)
    print("Head initialized with imprinted weights.")