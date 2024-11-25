import torch
import torch.nn as nn
from ..attack import Attack

class UniversalLatentSpaceAttack(Attack):
    def __init__(self, vae_model, classifier, latent_dim, batch_size, steps=1000, alpha=0.01, target_class=None):
        """
        Universal Latent-space attack to generate a universal adversarial perturbation in StyleGAN's latent space.
        [https://arxiv.org/abs/2004.00622]

        Args:
            vae_model (nn.Module): Pre-trained VAE model with encoder and decoder.
            classifier (nn.Module): Forensic classifier to attack, f(x).
            latent_dim_w (int): Dimensionality of the StyleGAN low-level latent space W.
            steps (int): Number of gradient descent steps. Default is 1000.
            alpha (float): Learning rate for optimization. Default is 0.01.
            target_class (int, optional): Class to misclassify as. If None, untargeted attack. Default is None.
        """
        super().__init__("UniversalLatentSpaceAttack", classifier)
        self.classifier = classifier
        self.vae_model = vae_model
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.steps = steps
        self.alpha = alpha
        self.target_class = target_class
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the universal perturbation z_tilde
        self.z_tilde = torch.randn((self.batch_size, self.latent_dim), device=self.device, requires_grad=True)

    def forward(self, images, labels):
        optimizer = torch.optim.Adam([self.z_tilde], lr=self.alpha)
        loss_fn = nn.CrossEntropyLoss()

        # Encode the images into latent space
        with torch.no_grad():
            latent_encoding = self.vae_model.encode(images)  # Single tensor output
            latent_mean = latent_encoding.latent_dist.mean  # Use .latent_dist() to extract latent representations if applicable
            latent_logvar = latent_encoding.latent_dist.logvar  # Simulate log variance if necessary
            z = latent_mean + torch.exp(0.5 * latent_logvar) * torch.randn_like(latent_mean)

        for _ in range(self.steps):
            # Add the universal perturbation to the latent vector
            z_adv = z + self.z_tilde

            # Decode the perturbed latent vector back to an image
            adversarial_image = self.vae_model.decode(z_adv)

            # Pass the adversarial image through the classifier
            outputs = self.classifier(adversarial_image)

            # Use target labels if specified, otherwise use true labels
            if self.target_class is not None:
                labels = torch.full_like(labels, self.target_class)

            # Compute the loss and optimize
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Optionally, clip the perturbation values to keep them valid
            self.z_tilde.data.clamp_(-1, 1)

        # Generate final adversarial image
        z_adv = z + self.z_tilde
        adversarial_image = self.vae_model.decode(z_adv)

        return adversarial_image