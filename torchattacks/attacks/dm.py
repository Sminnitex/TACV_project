import torch
import torch.nn as nn
from ..attack import Attack

class DistortionMinimizingAttack(Attack):
    r"""
    Distortion Minimizing Attack: An attack that minimizes the perturbation, specifically 
    with a p-norm constraint (e.g., L2), and ensuring that the perturbed image is misclassified.
    [https://arxiv.org/abs/2004.00622]

    Args:
            model (nn.Module): The classifier to attack.
            threshold (float): The threshold value below which the classifier should classify as real.
            norm_type (int): The p-norm to use for the perturbation (e.g., 2 for L2, 1 for L1, etc.).
            epsilon (float): The maximum allowed perturbation per pixel (e.g., 1/255 for the LSB).
            max_steps (int): The number of iterations for gradient descent.
            alpha (float): The learning rate for optimization.
            c0 (float): Lower bound of the hyperparameter c range.
            c1 (float): Upper bound of the hyperparameter c range.
            binary_search_steps (int): The number of steps to perform the binary search.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height`, `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.DistortionMinimizingAttack(model, threshold=0.5, norm_type=2, epsilon=1/255, max_steps=1000, alpha=0.01, c0=0, c1=100)
        >>> adv_images = attack(images, labels)
    """
    
    def __init__(self, model, threshold=0.5, norm_type=2, epsilon=1/255, max_steps=1000, alpha=0.01, c0=0, c1=100, binary_search_steps=10):
        super().__init__("DistortionMinimizingAttack", model)
        self.threshold = threshold
        self.norm_type = norm_type  # L2, L1, etc.
        self.epsilon = epsilon  # Max perturbation per pixel
        self.max_steps = max_steps
        self.alpha = alpha
        self.c0 = c0  # Lower bound of the c range
        self.c1 = c1  # Upper bound of the c range
        self.binary_search_steps = binary_search_steps  # Number of binary search steps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, image, label):
        image = image.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device).float()

        # Binary search for optimal c
        c0, c1 = self.c0, self.c1
        for _ in range(self.binary_search_steps):
            c = (c0 + c1) / 2
            success, _= self.run_attack(image, label, c)
            
            if success:
                c0 = c  # If attack succeeds, search in the upper half
            else:
                c1 = c  # If attack fails, search in the lower half

        # After binary search, run the attack with the found optimal c
        optimal_c = (c0 + c1) / 2
        _, adversarial_image = self.run_attack(image, label, optimal_c)

        return adversarial_image

    def run_attack(self, image, label, c):
        """
        Performs the attack for a specific value of the hyperparameter c.
        
        Args:
            image (torch.Tensor): The input image.
            label (torch.Tensor): The target label for classification.
            c (float): The hyperparameter controlling the trade-off between perturbation size and attack success.

        Returns:
            bool: Whether the attack was successful.
            adversarial_image (torch.Tensor): The generated adversarial image.
        """
        # Initialize the perturbation δ (initially set to 0)
        delta = torch.zeros_like(image, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=self.alpha)
        loss_fn = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss for real/fake classification

        # Run optimization to generate the perturbation
        for _ in range(self.max_steps):
            perturbed_image = image + delta
            perturbed_image = torch.clamp(perturbed_image, 0, 1)  # Ensure the image stays within valid range [0, 1]

            # Classifier output
            output = self.get_logits(perturbed_image)
            output_fake = output[:, 1]
            loss = loss_fn(output_fake, label)

            # Add the L2 regularization to the loss function, scaled by c
            loss += c * torch.norm(delta, p=self.norm_type)

            # Perform backpropagation and update the perturbation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Project the perturbation back into the valid range (satisfy the perturbation size constraint)
            delta.data = self.project_onto_norm_ball(delta.data)

        # Final adversarial image
        adversarial_image = image + delta
        adversarial_image = torch.clamp(adversarial_image, 0, 1)  # Ensure valid pixel range

        # Check if the attack was successful (misclassification)
        success = self.is_successful(perturbed_image, label)
        return success, adversarial_image

    def project_onto_norm_ball(self, delta):
        """
        Projects the perturbation onto the p-norm ball with maximum perturbation per pixel.
        
        Args:
            delta (torch.Tensor): The perturbation tensor.
        
        Returns:
            torch.Tensor: The projected perturbation, clipped to satisfy the norm constraint.
        """
        # Ensure the perturbation does not exceed the maximum allowed epsilon per pixel
        norm = torch.norm(delta.view(delta.size(0), -1), self.norm_type, dim=1, keepdim=True)
        factor = torch.min(torch.ones_like(norm), self.epsilon / norm)
        delta = delta * factor.view(-1, 1, 1, 1)  # Scale the perturbation by the factor

        return delta

    def is_successful(self, perturbed_image, label):
        """
        Determines whether the attack was successful by checking if the classifier misclassifies the perturbed image.
        
        Args:
            perturbed_image (torch.Tensor): The perturbed image.
            label (torch.Tensor): The target label.
        
        Returns:
            bool: Whether the attack was successful.
        """
        output = self.get_logits(perturbed_image)
        predicted_label = torch.argmax(output, dim=1)
        return (predicted_label != label).all().item()  # Misclassification success