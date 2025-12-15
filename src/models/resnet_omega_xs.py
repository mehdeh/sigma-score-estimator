"""
ResNet model for estimating omega(x, sigma) - noise-level score gradient conditioned on sigma.
This model takes both the image and noise level as input and outputs a single scalar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for ResNet."""
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        # If the stride or the number of input/output channels differs, use a shortcut
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the shortcut connection
        out = F.relu(out)
        return out


class ResNetOmegaXSigma(nn.Module):
    """
    ResNet18-based model for estimating omega(x, sigma).
    
    This model estimates the noise-level score gradient for a given noisy image
    conditioned on the noise level sigma.
    
    Args:
        block: The residual block class to use (e.g., BasicBlock)
        num_blocks: List of integers specifying the number of blocks in each layer
    
    Input:
        x: RGB image tensor of shape (batch_size, 3, 32, 32)
        sigma: Noise level tensor of shape (batch_size,)
    
    Output:
        Scalar tensor of shape (batch_size, 1) representing omega(x, sigma)
    """
    
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2]):
        super(ResNetOmegaXSigma, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Embedding layer to combine image features and sigma scalar
        self.fc_embedding = nn.Linear(512 * block.expansion + 1, 512)

        # Final regression output
        self.fc_out = nn.Linear(512, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, sigma):
        """
        Forward pass through the network.
        
        Args:
            x: Input image tensor of shape (batch_size, 3, 32, 32)
            sigma: Noise level tensor of shape (batch_size,) or (batch_size, 1)
        
        Returns:
            omega: Scalar output tensor of shape (batch_size, 1)
        """
        # Process the image input through ResNet layers
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Apply global average pooling
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)  # Flatten the output

        # Ensure sigma is of shape (batch_size, 1)
        sigma = sigma.view(sigma.size(0), 1)
        
        # Concatenate the scalar sigma with the features from the image
        combined_input = torch.cat((out, sigma), dim=1)

        # Pass the combined input through the embedding layer
        # Note: No ReLU activation here to match the original draft implementation
        embedded = self.fc_embedding(combined_input)

        # Final regression output
        output = self.fc_out(embedded)
        return output


def resnet18_omega_xs():
    """
    Create a ResNet18-based model for omega(x, sigma) estimation.
    
    Returns:
        ResNetOmegaXSigma model instance
    """
    return ResNetOmegaXSigma(BasicBlock, [2, 2, 2, 2])

