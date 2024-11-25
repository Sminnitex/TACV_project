import torch
import torch.nn as nn
from ..attack import Attack

class BlackBoxTransferAttack(Attack):
    def __init__(self, source_classifier, target_classifier, epsilon=1/255, max_steps=1000, alpha=0.01):
        """
        Black-box adversarial attack exploiting transferability of adversarial examples.

        Args:
            source_classifier (nn.Module): The classifier to generate adversarial examples against.
            target_classifier (nn.Module): The forensic classifier that will be attacked.
            epsilon (float): Maximum allowed perturbation per pixel.
            max_steps (int): Number of gradient descent iterations for attack optimization.
            alpha (float): Learning rate for gradient descent.
        """
        super().__init__("BlackBoxTransferAttack", source_classifier)
        self.source_classifier = source_classifier
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.alpha = alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.target_classifier = target_classifier.to(self.device)

    def forward(self, image, label):
        """
        Apply the adversarial attack to a single image, optimizing perturbation for transferability.

        Args:
            image (torch.Tensor): The input image (either real or fake).
            label (torch.Tensor): The true label for the image (real or fake).

        Returns:
            torch.Tensor: The adversarial image after perturbation.
        """
        image = image.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device).float()

        # Initialize the perturbation δ (initially set to 0)
        delta = torch.zeros_like(image, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([delta], lr=self.alpha)
        loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for real/fake classification

        for _ in range(self.max_steps):
            # Perturbed image
            perturbed_image = image + delta
            perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure the image stays within valid range [0, 1]

            # Classifier output for the source model (white-box attack)
            source_output = self.source_classifier(perturbed_image)
            source_output_fake = source_output[:, 1]
            source_loss = loss_fn(source_output_fake, label)

            # We want to misclassify the image as real (label = 0)
            source_loss = -source_loss  # Invert the loss for adversarial generation

            # Classifier output for the target model (black-box attack)
            target_output = self.target_classifier(perturbed_image)
            target_output_fake = target_output[:, 1]
            target_loss = loss_fn(target_output_fake, label)

            # Total loss: we want to minimize source and target loss, and the perturbation norm
            total_loss = source_loss + target_loss

            # Add the L2 regularization to the loss function for the perturbation
            total_loss += torch.norm(delta, p=2)

            # Perform backpropagation and update the perturbation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Project the perturbation back into the valid range
            delta.data = self.project_onto_norm_ball(delta.data)

        # Final adversarial image
        adversarial_image = image + delta
        adversarial_image = torch.clamp(adversarial_image, 0, 1)  # Ensure valid pixel range

        return adversarial_image

    def project_onto_norm_ball(self, delta):
        """
        Project the perturbation onto the p-norm ball with maximum perturbation per pixel.
        
        Args:
            delta (torch.Tensor): The perturbation tensor.
        
        Returns:
            torch.Tensor: The projected perturbation, clipped to satisfy the norm constraint.
        """
        # Ensure the perturbation does not exceed the maximum allowed epsilon per pixel
        norm = torch.norm(delta.view(delta.size(0), -1), 2, dim=1, keepdim=True)
        factor = torch.min(torch.ones_like(norm), self.epsilon / norm)
        delta = delta * factor.view(-1, 1, 1, 1)  # Scale the perturbation by the factor

        return delta