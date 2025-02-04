import os
from multiprocessing import cpu_count

# General OpenMP settings
default_threads = max(1, int(cpu_count() * 0.75))
os.environ["OMP_NUM_THREADS"] = str(default_threads)

# Faiss specific optimizations
os.environ["OPENBLAS_NUM_THREADS"] = str(default_threads)
os.environ["MKL_NUM_THREADS"] = str(default_threads)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(default_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(default_threads)

import faiss
import numpy as np
from pathlib import Path
import torch
import open_clip
from PIL import Image
import json
from typing import List, Tuple, Union
from llm_utils import get_location_from_query
from image_utils import get_gps_data, haversine_distance

class FaissImageIndex:
    def __init__(self, dimension=512, cache_dir="faiss_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index_file = self.cache_dir / "faiss.index"
        self.metadata_file = self.cache_dir / "metadata.json"
        self.dimension = dimension
        self.load_or_create_index()
    
    def load_or_create_index(self):
        if self.index_file.exists() and self.metadata_file.exists():
            self.index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.clear_index()
    
    def clear_index(self):
        """Reset the index to empty state"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = {'paths': [], 'sizes': []}
        self.save()
    
    def contains_path(self, path: str) -> bool:
        """Check if path is already in index"""
        return str(path) in self.metadata['paths']
    
    def add_image(self, path: Path, features: np.ndarray, size: Tuple[int, int]):
        features = features.astype('float32')
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        self.index.add(features)
        self.metadata['paths'].append(str(path))
        self.metadata['sizes'].append(size)
        self.save()
    
    def save(self):
        faiss.write_index(self.index, str(self.index_file))
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f)
    
    def search(self, query_features: np.ndarray, k: int = 100, min_score: float = 20.0):
        if query_features.ndim == 1:
            query_features = query_features.reshape(1, -1)
        query_features = query_features.astype('float32')
        
        scores, indices = self.index.search(query_features, k)
        results = []
        
        for idx, score in zip(indices[0], scores[0]):
            if score * 100 >= min_score:  # Convert to percentage
                results.append({
                    'path': self.metadata['paths'][idx],
                    'size': self.metadata['sizes'][idx],
                    'mean_score': float(score * 100)
                })
        
        return results

    def verify_index(self):
        """Verify index integrity and clean up if needed"""
        valid_paths = []
        valid_sizes = []
        
        for path, size in zip(self.metadata['paths'], self.metadata['sizes']):
            if Path(path).exists():
                valid_paths.append(path)
                valid_sizes.append(size)
            else:
                print(f"Removing missing file from index: {path}")
                
        if len(valid_paths) != len(self.metadata['paths']):
            print("Rebuilding index with only valid files...")
            features = []
            for i in range(self.index.ntotal):
                if i < len(valid_paths):
                    features.append(self.index.reconstruct(i))
            
            # Create new index
            self.index = faiss.IndexFlatIP(self.dimension)
            if features:
                self.index.add(np.stack(features))
            
            self.metadata = {
                'paths': valid_paths,
                'sizes': valid_sizes
            }
            self.save()
            
        return len(valid_paths)

_model = None
_preprocess = None

def get_model():
    """Get or initialize the CLIP model"""
    global _model, _preprocess
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        _model = _model.cuda()
    return _model, _preprocess

def find_similar_to_images_faiss(
    references: List[Tuple[str, float]],
    directory: str,
    batch_size: int = 768,
    min_score: float = 20.0,
    min_dimensions: Tuple[int, int] = (300, 300),
    cache_dir: str = "faiss_cache"
):
    model, preprocess = get_model()
    index = FaissImageIndex(cache_dir=cache_dir)

    # Process reference images
    reference_features = []
    reference_weights = []
    print("Processing reference images...")
    for ref_path, weight in references:
        try:
            reference_img = preprocess(Image.open(ref_path)).unsqueeze(0).cuda()
            with torch.no_grad():
                features = model.encode_image(reference_img)
                features /= features.norm(dim=-1, keepdim=True)
                reference_features.append(features)
                reference_weights.append(weight)
        except Exception as e:
            print(f"Error processing reference image {ref_path}: {e}")
            continue
    
    if not reference_features:
        return []
    
    reference_features = torch.cat(reference_features, dim=0)
    reference_weights = torch.tensor(reference_weights).cuda()
    
    # Find all images
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    reference_paths = {str(Path(ref[0])) for ref in references}
    
    for ext in image_extensions:
        image_files.extend(Path(directory).rglob(f'*{ext}'))
        image_files.extend(Path(directory).rglob(f'*{ext.upper()}'))
    
    # Filter out references and already indexed images
    total_images = len(set(image_files))
    image_files = [f for f in set(image_files) 
                  if str(f) not in reference_paths 
                  and not index.contains_path(f)]
    
    print(f"Found {total_images} total images")
    print(f"Skipping {total_images - len(image_files)} already indexed images")
    print(f"Processing {len(image_files)} new images...")
    
    # Process new images in batches
    total_processed = 0
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        valid_files = []
        
        for img_path in batch_files:
            try:
                image = Image.open(img_path)
                if image.size[0] >= min_dimensions[0] and image.size[1] >= min_dimensions[1]:
                    image_tensor = preprocess(image)
                    batch_images.append(image_tensor)
                    valid_files.append((img_path, image.size))
            except Exception as e:
                print(f"Error with {img_path}: {e}")
                continue
        
        if not batch_images: continue
        
        try:
            with torch.no_grad():
                image_tensor = torch.stack(batch_images).cuda()
                features = model.encode_image(image_tensor)
                features /= features.norm(dim=-1, keepdim=True)
                
                for idx, (path, size) in enumerate(valid_files):
                    index.add_image(path, features[idx].cpu().numpy(), size)
                
                total_processed += len(valid_files)
                print(f"Processed {total_processed}/{len(image_files)} new images...")
                
        except RuntimeError as e:
            print(f"Error processing batch: {e}")
            continue
        
        if i % (batch_size * 4) == 0:
            torch.cuda.empty_cache()
    
    print("Searching for similar images...")
    weighted_results = []
    for ref_features, weight in zip(reference_features, reference_weights):
        results = index.search(ref_features.cpu().numpy(), k=100, min_score=min_score)
        for result in results:
            result['individual_scores'] = {f'ref_{len(weighted_results)}': result['mean_score']}
            result['mean_score'] *= weight.item()
            weighted_results.append(result)
    
    weighted_results.sort(key=lambda x: x['mean_score'], reverse=True)
    seen_paths = set()
    final_results = []
    for result in weighted_results:
        if result['path'] not in seen_paths:
            seen_paths.add(result['path'])
            final_results.append(result)
    
    print(f"Found {len(final_results)} similar images")
    return final_results[:100]


async def find_similar_images_faiss(
    root_dir: Union[str, Path],
    weighted_queries: List[Tuple[str, float]],
    batch_size: int = 768,
    min_score: float = 20.0,
    min_dimensions: Tuple[int, int] = (500, 500),
    cache_dir: str = "faiss_cache",
    use_location: bool = True
):
    print("Initializing models...")
    model, preprocess = get_model()
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    index = FaissImageIndex(cache_dir=cache_dir)
    
    # Check all queries for locations
    location_infos = []

    if use_location:
        for query, weight in weighted_queries:
            try:
                loc_data = await get_location_from_query(query)
                if loc_data and loc_data.get('has_location', False):
                    loc_data['weight'] = weight
                    location_infos.append(loc_data)
                    print(f"Found location in query '{query}': {loc_data['location_name']}")
            except Exception as e:
                print(f"Warning: Location service error for query '{query}': {e}")
                continue
    
    # Process text queries
    print("Processing text queries...")
    queries, weights = zip(*weighted_queries)
    weights = torch.tensor(weights).cuda()
    text_tokens = tokenizer(queries).cuda()
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Find all images
    print("Scanning directory for images...")
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(root_dir).rglob(f'*{ext}'))
        image_files.extend(Path(root_dir).rglob(f'*{ext.upper()}'))
    
    # Filter out already indexed images
    total_images = len(set(image_files))
    image_files = [f for f in set(image_files) if not index.contains_path(f)]
    
    print(f"Found {total_images} total images")
    print(f"Skipping {total_images - len(image_files)} already indexed images")
    print(f"Processing {len(image_files)} new images...")
    
    # Process images in batches
    total_processed = 0
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        batch_images = []
        valid_files = []
        
        for img_path in batch_files:
            try:
                image = Image.open(img_path)
                if image.size[0] >= min_dimensions[0] and image.size[1] >= min_dimensions[1]:
                    image_tensor = preprocess(image)
                    batch_images.append(image_tensor)
                    valid_files.append((img_path, image.size))
            except Exception as e:
                print(f"Error with {img_path}: {e}")
                continue
        
        if not batch_images: continue
        
        try:
            with torch.no_grad():
                image_tensor = torch.stack(batch_images).cuda()
                features = model.encode_image(image_tensor)
                features /= features.norm(dim=-1, keepdim=True)
                
                for idx, (path, size) in enumerate(valid_files):
                    index.add_image(path, features[idx].cpu().numpy(), size)
                
                total_processed += len(valid_files)
                print(f"Processed {total_processed}/{len(image_files)} new images...")
                
        except RuntimeError as e:
            print(f"Error processing batch: {e}")
            continue
        
        if i % (batch_size * 4) == 0:
            torch.cuda.empty_cache()
    
    print("Searching with text queries...")
    weighted_results = []
    for query, query_features, weight in zip(queries, text_features, weights):
        print(f"Searching for: '{query}'")
        results = index.search(query_features.cpu().numpy(), k=100, min_score=min_score)
        
        for result in results:
            try:
                base_score = float(result['mean_score']) * float(weight.item())
                location_scores = {}

                # Check each location if any were found
                if location_infos:
                    img_gps = get_gps_data(result['path'])
                    if img_gps:
                        for loc_info in location_infos:
                            try:
                                distance = haversine_distance(
                                    float(loc_info['latitude']),
                                    float(loc_info['longitude']),
                                    float(img_gps['latitude']),
                                    float(img_gps['longitude'])
                                )
                                max_distance = 100  # km
                                location_score = max(0, 100 * (1 - distance/max_distance))
                                location_name = loc_info['location_name']
                                loc_weight = float(loc_info.get('weight', 1.0))
                                
                                # Store both score and distance for each location
                                location_scores[f"location_{location_name}"] = {
                                    'score': location_score,
                                    'distance': distance
                                }
                                # Adjust base score based on location match
                                base_score = 0.7 * base_score + 0.3 * location_score * loc_weight
                            except (TypeError, ValueError) as e:
                                print(f"Error processing location {loc_info['location_name']}: {e}")
                                continue
                
                result['mean_score'] = base_score
                result['individual_scores'] = {
                    query: float(base_score),
                    **location_scores
                }
                weighted_results.append(result)
            except (TypeError, ValueError) as e:
                print(f"Error processing result: {e}")
                continue
    
    # Sort and return results
    weighted_results.sort(key=lambda x: x['mean_score'], reverse=True)
    seen_paths = set()
    final_results = []
    for result in weighted_results:
        if result['path'] not in seen_paths:
            seen_paths.add(result['path'])
            final_results.append(result)
    
    return final_results[:100]

def find_nearby_similar_images(
    reference_path: str,
    max_distance_km: float = 10.0,
    min_similarity: float = 20.0,
    cache_dir: str = "faiss_cache"
) -> List[dict]:
    # Get GPS data of reference image
    ref_gps = get_gps_data(reference_path)
    if not ref_gps:
        print(f"No GPS data found in reference image: {reference_path}")
        return []
    
    print(f"Reference GPS: {ref_gps}")
    
    # Get visual features of reference image
    model, preprocess = get_model()
    index = FaissImageIndex(cache_dir=cache_dir)
    
    try:
        ref_image = Image.open(reference_path)
        if ref_image.mode != 'RGB':
            ref_image = ref_image.convert('RGB')
        ref_tensor = preprocess(ref_image).unsqueeze(0).cuda()
        
        with torch.no_grad():
            ref_features = model.encode_image(ref_tensor)
            ref_features /= ref_features.norm(dim=-1, keepdim=True)
            
        # Find visually similar images from index
        similar_results = index.search(ref_features.cpu().numpy(), k=500, min_score=min_similarity)
        print(f"Found {len(similar_results)} visually similar images in index")
        
        # Filter by distance and combine scores
        final_results = []
        for result in similar_results:
            img_path = result['path']
            if str(img_path) == str(reference_path):
                continue
                
            img_gps = get_gps_data(img_path)
            if img_gps:
                distance = haversine_distance(
                    ref_gps['latitude'],
                    ref_gps['longitude'],
                    img_gps['latitude'],
                    img_gps['longitude']
                )
                if distance <= max_distance_km:
                    distance_score = 100 * (1 - distance/max_distance_km)
                    combined_score = 0.7 * result['mean_score'] + 0.3 * distance_score
                    
                    final_results.append({
                        'path': img_path,
                        'distance_km': distance,
                        'visual_score': result['mean_score'],
                        'distance_score': distance_score,
                        'combined_score': combined_score,
                        'individual_scores': {'visual': result['mean_score']},
                        'coordinates': img_gps
                    })
        
        print(f"Found {len(final_results)} nearby similar images")
        final_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return final_results
        
    except Exception as e:
        print(f"Error processing reference image: {e}")
        return []
