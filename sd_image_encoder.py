from diffusers import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from typing import Union, List, Tuple
from pathlib import Path

class SDImageEncoder:
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").cuda()
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").cuda()
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        
    def encode_text(self, text):
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        input_ids = tokens.input_ids.cuda()
        text_embeddings = self.text_encoder(input_ids)[0]
        embeddings = text_embeddings.mean(dim=1)
        # Normalize embeddings
        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        return embeddings

    def encode_image(self, image):
        # Image is already a tensor, just make sure it's on cuda and batched
        if isinstance(image, torch.Tensor):
            if image.dim() == 3:
                image = image.unsqueeze(0)
            image = image.cuda()
        else:
            # Handle numpy array input
            image = torch.from_numpy(image).cuda().unsqueeze(0)
            
        # Get latent representation
        latent = self.vae.encode(image).latent_dist.mode()
        
        # Normalize latent representation
        latent = latent.reshape(-1, 512)
        latent = latent / latent.norm(dim=-1, keepdim=True)
        return latent


def find_similar_images_sd(
    root_dir: Union[str, Path],
    weighted_queries: List[Tuple[str, float]],
    batch_size: int = 768,
    min_score: float = 20.0,
    cache_dir: str = "sd_cache"
):
    encoder = SDImageEncoder()
    index = FaissImageIndex(cache_dir=cache_dir)
    
    # Process text queries
    queries, weights = zip(*weighted_queries)
    text_features = []
    
    with torch.no_grad():
        for query in queries:
            text_feature = encoder.encode_text(query)
            text_features.append(text_feature)
    
    text_features = torch.stack(text_features)
    weights = torch.tensor(weights).cuda()
    
    # Process images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(Path(root_dir).rglob(f'*{ext}'))
    
    # Process in batches
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_features = []
        valid_files = []
        
        with torch.no_grad():
            for img_path in batch_files:
                try:
                    image = Image.open(img_path)
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    # Get latent representation
                    latent = encoder.encode_image(image)
                    batch_features.append(latent)
                    valid_files.append((img_path, image.size))
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
                    continue
            
            if batch_features:
                features = torch.stack(batch_features)
                
                # Add to index
                for idx, (path, size) in enumerate(valid_files):
                    index.add_image(path, features[idx].cpu().numpy(), size)
