import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super(MyModel, self).__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),  
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01),  # Replace ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 112x112

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.01),  # Replace ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 56x56

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.01),  # Replace ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 28x28
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.01),  # Replace ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 14x14
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.01),  # Replace ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 7x7
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),             
            nn.Linear(512 * 7 * 7, 2048),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)

        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
