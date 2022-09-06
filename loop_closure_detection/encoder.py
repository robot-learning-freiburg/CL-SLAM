import torch
from torch import Tensor
from torchvision import models, transforms
from torchvision.models.feature_extraction import create_feature_extractor


class FeatureEncoder:
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        # self.model = models.inception_v3(pretrained=True, transform_input=False)
        self.model = models.mobilenet_v3_small(pretrained=True)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # self.model.to(self.device)
        # self.model.eval()

        # train_nodes, eval_nodes = get_graph_node_names(self.model)
        # pprint(eval_nodes)s

        self.model = create_feature_extractor(self.model, return_nodes=['flatten'])
        self.num_features = 576

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, image: Tensor) -> Tensor:
        image = self.normalize(image)
        image = image.to(self.device)
        with torch.no_grad():
            features = self.model(image)['flatten']
        return features
