import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights
from torch import nn, optim
from sklearn.metrics import accuracy_score
from PIL import Image
from tqdm import tqdm
from torchattacks import PGD, DistortionMinimizingAttack, LossMaximizingAttack, UniversalAdversarialAttack, UniversalLatentSpaceAttack, BlackBoxTransferAttack
from transformers import CLIPModel, AutoModel, CLIPImageProcessor
from huggingface_hub import login
from diffusers.models import AutoencoderKL

#Dataset custom class
class CustomDataset(Dataset):
    def __init__(self, image_dir, label, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.label = label
        self.images = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir) if fname.endswith(('png', 'jpg', 'jpeg'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.label

#Get prediction    
def get_pred(model, images, device):
    logits = model(images.to(device))
    _, pres = logits.max(dim=1)
    return pres.cpu()

#Set basic stuff
torch.cuda.empty_cache()
login(token="ifneeded")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_dir = os.path.join(os.getcwd(), "synthbuster")
save_path = "./weights/resnet_weights.pt"

#Transformations: resizing, cropping, normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
])

#Create dataset
real_path = os.path.join(dataset_dir, "real_RAISE_1k")
fake_folders = [
    "dalle2", "dalle3", "firefly", "glide", "midjourney-v5",
    "stable-diffusion-1-3", "stable-diffusion-1-4", "stable-diffusion-2", "stable-diffusion-xl"
]
fake_paths = [os.path.join(dataset_dir, folder) for folder in fake_folders]

real_dataset = CustomDataset(image_dir=real_path, label=0, transform=transform)
fake_datasets = [CustomDataset(image_dir=path, label=1, transform=transform) for path in fake_paths] 
fake_dataset = ConcatDataset(fake_datasets)
dataset = ConcatDataset([real_dataset, fake_dataset]) 

#Split into training and testing datasets (80-20 split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#Data loaders
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#Load pretrained model
weights = ResNet50_Weights.IMAGENET1K_V1
model = resnet50(weights=weights)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float32)
latent_dim = vae.config.latent_channels
transfer_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

#Modify the final layer for binary classification (real vs fake) 
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)
transfer_model.classifier = nn.Linear(transfer_model.classifier[1].in_features, 2)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    predictions = []
    train_labels = []
    
    with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as pbar:
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = accuracy_score(train_labels, predictions)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}") 
torch.save(model.state_dict(), save_path)
print(f"Model weights saved to {save_path}")

#Testing loop
model.load_state_dict(torch.load(save_path, weights_only=False))
print(f"Model weights loaded from {save_path}")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    with tqdm(total=len(test_loader), desc="Testing", unit="batch") as pbar:
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            preds = get_pred(model, images, device)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.update(1)
accuracy = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {accuracy:.4f}") 

#Adversarial attacks
attacks = [
    ("PGD", PGD(model), "_pgd.pt"),
    ("Distortion Minimizing", DistortionMinimizingAttack(model, threshold=0.5, norm_type=2, epsilon=1/255, max_steps=6, alpha=0.5, c0=0, c1=100, binary_search_steps=7), "_dm.pt"),
    ("Loss Maximizing", LossMaximizingAttack(model, norm_type=2, epsilon=0.1, steps=15, alpha=0.5), "_lm.pt"),
    ("Universal Patch", UniversalAdversarialAttack(model, patch_size=(0.01, 0.01), steps=15, alpha=0.5, target_class=None), "_patch.pt"),
    #("Universal Latent", UniversalLatentSpaceAttack(vae, model, latent_dim=latent_dim, batch_size=batch_size, steps=10, alpha=0.01, target_class=None), "_ulsa.pt"),
    #("Black Box Transfer", BlackBoxTransferAttack(model, target_classifier=transfer_model, threshold=0.5, norm_type=2, epsilon=1/255, max_steps=5, alpha=0.01, c0=0, c1=100, binary_search_steps=5), "_blackBox.pt"),
]

#Iterate through each attack
for attack_name, attack, save_path in attacks:
    print(f"Running {attack_name} Attack")

    #Test on adversarial images
    attack.save(data_loader=test_loader, save_path=save_path)
    adv_loader = attack.load(load_path=save_path)

    model.eval()
    adv_all_preds = []
    adv_all_labels = []

    with tqdm(total=len(adv_loader), desc=f"{attack_name} Attack", unit="batch") as pbar:
        for adv_images, labels in adv_loader:
            preds = get_pred(model, adv_images, device)
            adv_all_preds.extend(preds.cpu().numpy())
            adv_all_labels.extend(labels.cpu().numpy())

            pbar.update(1)

    #Calculate adversarial accuracy
    adv_accuracy = accuracy_score(adv_all_labels, adv_all_preds)
    print(f"{attack_name} Adversarial Accuracy: {adv_accuracy:.4f}\n")
