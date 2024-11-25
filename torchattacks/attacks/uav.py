import torch
import torch.nn as nn
from ..attack import Attack

class UniversalAdversarialAttack(Attack):
    r"""
    Universal Adversarial Patch Attack: Generates a universal noise patch that, when applied to any input image,
    causes the image to be misclassified. This approach is efficient as a single patch can be reused across multiple inputs.
    [https://arxiv.org/abs/2004.00622]

    Arguments:
        model (nn.Module): Model to attack.
        patch_size (tuple): Size of the patch as a fraction of the input image dimensions (height, width).
        steps (int): Number of gradient descent steps to generate the patch. (Default: 1000)
        alpha (float): Learning rate for gradient descent. (Default: 0.01)
        target_class (int): Target class index for misclassification. Use `None` for untargeted attack. (Default: None)

    Shape:
        - patch: :math:`(1, C, H, W)` where `C` is the number of channels, `H` and `W` are determined by the `patch_size` relative to input dimensions.
        - images: :math:`(N, C, H, W)` where `N` is batch size.
        - labels: :math:`(N)` for the original labels.

    Examples::
        >>> attack = torchattacks.UniversalAdversarialPatch(model, patch_size=(0.1, 0.1), steps=1000, alpha=0.01, target_class=None)
        >>> patch = attack.generate_patch()
        >>> adv_images = attack.apply_patch(images, patch)
    """
    
    def __init__(self, model, patch_size=(0.01, 0.01), steps=1000, alpha=0.01, target_class=None):
        super().__init__("UniversalAdversarialAttack", model)
        self.patch_size = patch_size
        self.steps = steps
        self.alpha = alpha
        self.target_class = target_class
        self.universal_patch = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, images, labels):
        # Train the universal patch if not already created
        if self.universal_patch is None:
            images = images.clone().detach().to(self.device)
            labels = labels.clone().detach().to(self.device)

            _, C, H, W = images.shape
            patch_H, patch_W = int(self.patch_size[0] * H), int(self.patch_size[1] * W)
            patch = torch.rand((1, C, patch_H, patch_W), device=self.device, requires_grad=True)

            optimizer = torch.optim.Adam([patch], lr=self.alpha)
            loss_fn = nn.CrossEntropyLoss()

            for _ in range(self.steps):
                # Targeted attack handling
                if self.target_class is not None:
                    labels = torch.full_like(labels, self.target_class)

                # Create adversarial images by overlaying the patch
                adv_images = self.apply_patch(images, patch)

                # Forward pass
                outputs = self.get_logits(adv_images)

                # Calculate loss
                if self.target_class is not None:
                    target_scores = outputs[:, self.target_class]
                    loss = -target_scores.mean()  # Maximize target class probability
                else:
                    loss = loss_fn(outputs, labels)

                # Gradient descent step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Clip patch values to keep them within valid range [0, 1]
                patch.data.clamp_(0, 1)

            # Save the generated patch
            self.universal_patch = patch.detach()

        # Apply the trained universal patch to the images
        adv_images = self.apply_patch(images, self.universal_patch)
        return adv_images

    def apply_patch(self, images, patch):
        """
        Overlays the patch onto the input images.
        """
        images = images.clone().detach().to(self.device)
        _, C, H, W = images.shape
        patch_H, patch_W = patch.shape[-2:]
        
        # Random placement during training
        x_offset = torch.randint(0, H - patch_H + 1, (1,)).item()
        y_offset = torch.randint(0, W - patch_W + 1, (1,)).item()

        adv_images = images.clone()
        adv_images[:, :, x_offset:x_offset+patch_H, y_offset:y_offset+patch_W] = patch

        return torch.clamp(adv_images, 0, 1)
