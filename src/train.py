import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torchvision.models as model
import torchvision.transforms as transform



def main():
    # Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set DataLoader
    class DataLoader(data.Dataset):
        def __init__(self, data_path, json_path, transform=None):
            """
            :param data_path:
            :param json_path:
            :param transform:
            """

        def __getitem__(self, item):
            pass


        def __len__(self):
            pass

    # Define Model
    class CNNModel(nn.Module):
        def __int__(self):
            super(CNNModel, self).__init__()
            resent = model.resnet152(pretrained=True)
            module_list = list(resent.children())[:-1]
            self.resnet_module = nn.Sequential(*module_list)
            self.linear_layer = nn.Linear(resent.fc.in_features, 3)

        def forward(self, input_images):
            with torch.no_grad():
                resnet_features = self.resnet_module(input_images)
            resnet_features = resnet_features.reshape(resnet_features.size(0), -1)
            final_features = self.linear_layer(resnet_features)
            return F.log_softmax(final_features, dim=1)
