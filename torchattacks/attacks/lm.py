import torch
import torch.nn as nn
from ..attack import Attack

class LossMaximizingAttack(Attack):
    r"""
    Loss-Maximizing Attack: An attack that maximizes the loss for a fake image to be misclassified as real,
    while ensuring that the distortion (perturbation) remains within a given threshold.
    [https://arxiv.org/abs/2004.00622]

    Args:
            model (nn.Module): Classifier to attack.
            norm_type (int): The p-norm to use for the constraint (e.g., 2 for L2, 1 for L1 norm).
            epsilon (float): Maximum allowed perturbation in terms of p-norm.
            steps (int): Number of gradient descent steps.
            alpha (float): Learning rate for optimization.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height`, `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.LossMaximizing(model, norm_type=2, epsilon=0.1, steps=1000, alpha=0.01)
        >>> adv_images = attack(images, labels)
    """
    
    def __init__(self, model, norm_type=2, epsilon=0.1, steps=1000, alpha=0.01):
        super().__init__("LossMaximizingAttack", model)
        self.norm_type = norm_type  # L2, L1, etc.
        self.epsilon = epsilon
        self.steps = steps
        self.alpha = alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # Initialize perturbation δ as a tensor of zeros (same shape as images)
        delta = torch.zeros_like(images, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([delta], lr=self.alpha)
        loss_fn = nn.CrossEntropyLoss()

        for _ in range(self.steps):
            # Forward pass
            perturbed_images = images + delta
            perturbed_images = torch.clamp(perturbed_images, 0, 1)  # Clip to ensure valid pixel range

            # Get classifier output for the perturbed image
            outputs = self.model(perturbed_images)

            # Calculate the loss
            loss = loss_fn(outputs, labels)

            # Backpropagate the loss and update perturbation δ
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Project δ onto the constraint space (p-norm <= epsilon)
            delta.data = self.project_onto_norm_ball(delta.data)

        # Generate adversarial images after optimization
        adversarial_images = images + delta
        adversarial_images = torch.clamp(adversarial_images, 0, 1)  # Clip to valid pixel range

        return adversarial_images

    def project_onto_norm_ball(self, delta):
        """
        Projects the perturbation onto the p-norm ball with radius epsilon.
        
        Args:
            delta (torch.Tensor): The perturbation tensor.
        
        Returns:
            torch.Tensor: The projected perturbation, clipped to satisfy the p-norm constraint.
        """
        # Compute the p-norm of the perturbation
        norm = delta.norm(self.norm_type, dim=(1, 2, 3), keepdim=True)

        # If the norm is greater than epsilon, scale it down
        factor = torch.min(torch.ones_like(norm), self.epsilon / norm)
        delta = delta * factor

        return delta
