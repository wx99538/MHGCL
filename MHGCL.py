import torch
import torch.nn as nn
from DynamicalGraphLearning import DynamicalGraphLearning
from MultiViewHeteroGNNContrastiveLearning import MultiViewHeteroGNNContrastiveLearning
from CrossAttentionFusion import CrossAttentionFusion


class DHLMCLF(nn.Module):
    """
    Deep Heterogeneous Learning with Multi-View Contrastive Learning Framework (DHLMCLF)

    Combines three DynamicalGraphLearning modules with MultiViewHeteroGNNFusion for multi-modal learning.

    Args:
        feature_dim (list): List of input feature dimensions for each modality [dim1, dim2, dim3]
        in_dim (list): List of hidden layer dimensions for the GNNs
        class_num (int): Number of output classes
        k1 (int): Number of neighbors for graph construction in DynamicalGraphLearning
        k2 (int): Parameter for MultiViewHeteroGNNFusion
        k3 (int): Parameter for MultiViewHeteroGNNFusion
        dropout (float): Dropout probability
    """

    def __init__(self, feature_dim, in_dim, class_num, k1, k2, k3, dropout):
        super(DHLMCLF, self).__init__()

        # Initialize three Dynamical Graph Learning modules for each modality
        self.modality1_encoder = DynamicalGraphLearning(feature_dim[0], in_dim, class_num, k1, dropout)
        self.modality2_encoder = DynamicalGraphLearning(feature_dim[1], in_dim, class_num, k1, dropout)
        self.modality3_encoder = DynamicalGraphLearning(feature_dim[2], in_dim, class_num, k1, dropout)

        # Initialize multi-view fusion module
        self.align_and_fusion_module = MultiViewHeteroGNNContrastiveLearning([in_dim[-1], in_dim[-1]], k2, k3, dropout)
        # Final classifier
        self.classifier = nn.Linear(in_dim[-1], class_num)

        # Loss weighting parameters
        self.fusion_loss_weight = 0.1

    def forward(self, x1, x2, x3, y=None, testing=False):
        """
        Forward pass of the DHLMCLF model.

        Args:
            x1, x2, x3: Input features for three modalities
            y: Target labels (optional, required for training)
            testing: Boolean flag for test mode

        Returns:
            tuple: (predictions, total_loss) if not testing
                   (predictions, None) if testing
        """
        # Encode each modality
        x1, loss1 = self.modality1_encoder(x1, y, testing)
        x2, loss2 = self.modality2_encoder(x2, y, testing)
        x3, loss3 = self.modality3_encoder(x3, y, testing)

        # Align modalities
        fusion, align_and_fusion_loss = self.align_and_fusion_module(x1, x2, x3, y, testing)
        # fusion = torch.concatenate([x1, x2, x3], dim=1)
        # align_and_fusion_loss = 0
        # Make predictions
        predictions = self.classifier(fusion)

        # Calculate loss if in training mode
        if not testing:
            # Classification loss
            criterion = nn.CrossEntropyLoss()
            classification_loss = criterion(predictions, y)

            # Total loss (weighted sum)
            total_loss = (loss1 + loss2 + loss3) + 0.001*align_and_fusion_loss + classification_loss

            return predictions, total_loss

        return predictions, fusion