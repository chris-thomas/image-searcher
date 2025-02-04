from pathlib import Path
from typing import Union, List, Tuple
from fractions import Fraction

import numpy as np
import torch
from diffusers import AutoencoderKL
import open_clip
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from faiss_utils import get_model, FaissImageIndex
from llm_utils import get_location_from_query
from image_utils import get_gps_data, haversine_distance
from sd_image_encoder import SDImageEncoder

class HybridImageSearch:
    def __init__(self, clip_weight=0.6, sd_weight=0.4):
        self.clip_model, self.clip_preprocess = get_model()
        self.sd_encoder = SDImageEncoder()
        self.clip_weight = clip_weight
        self.sd_weight = sd_weight
        self.clip_index = None
        self.sd_index = None

    def normalize_scores(self, results, scores):
        """Normalize scores using percentile method"""
        if not results:
            return []
        
        scores = np.array(scores)
        p95 = np.percentile(scores, 95)
        p5 = np.percentile(scores, 5)
        
        normalized = []
        for result in results:
            if result and 'mean_score' in result and 'path' in result:
                score = (result['mean_score'] - p5) / (p95 - p5) * 100
                score = min(100.0, max(0.0, score))
                normalized.append({
                    'path': result['path'],
                    'score': score
                })
        return normalized

    def combine_scores(self, clip_scores, sd_scores, query, weight, location_scores=None):
        """Combine normalized scores from both models and location if available"""
        path_to_scores = {}
        
        # Process CLIP scores
        for result in clip_scores:
            path_to_scores[result['path']] = {
                'clip_score': result['score'],
                'sd_score': 0.0,
                'location_score': location_scores.get(result['path'], 0.0) if location_scores else 0.0
            }
        
        # Process SD scores
        for result in sd_scores:
            path = result['path']
            if path in path_to_scores:
                path_to_scores[path]['sd_score'] = result['score']
            else:
                path_to_scores[path] = {
                    'clip_score': 0.0,
                    'sd_score': result['score'],
                    'location_score': location_scores.get(path, 0.0) if location_scores else 0.0
                }
        
        # Calculate combined scores
        results = []
        for path, scores in path_to_scores.items():
            # Base score from CLIP and SD
            combined_score = (
                self.clip_weight * scores['clip_score'] +
                self.sd_weight * scores['sd_score']
            )
            
            # Add location influence if enabled
            if location_scores:
                combined_score = combined_score * 0.7 + scores['location_score'] * 0.3
            
            combined_score *= weight
            
            results.append({
                'path': path,
                'mean_score': combined_score,
                'individual_scores': {
                    f"{query} (CLIP)": scores['clip_score'],
                    f"{query} (SD)": scores['sd_score'],
                    f"{query} (Location)": scores['location_score'] if location_scores else 0.0
                }
            })
        
        return results

    def create_indices(self, cache_dir="search_cache"):
        self.clip_index = FaissImageIndex(cache_dir=f"{cache_dir}/clip", dimension=512)
        self.sd_index = FaissImageIndex(cache_dir=f"{cache_dir}/sd", dimension=512)

    def process_single_image(self, img_path, target_size, min_dimensions):
        """Process a single image for both CLIP and SD models"""
        try:
            Image.MAX_IMAGE_PIXELS = None
            image = Image.open(img_path)
            if image.size[0] < min_dimensions[0] or image.size[1] < min_dimensions[1]:
                return None
                
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize maintaining aspect ratio
            image.thumbnail(target_size, Image.Resampling.LANCZOS)
            
            # Add padding
            new_image = Image.new('RGB', target_size, (0, 0, 0))
            paste_pos = ((target_size[0] - image.size[0]) // 2,
                       (target_size[1] - image.size[1]) // 2)
            new_image.paste(image, paste_pos)
            
            clip_tensor = self.clip_preprocess(new_image)
            np_image = np.array(new_image).astype(np.float32) / 255.0
            sd_tensor = torch.from_numpy(np_image).permute(2, 0, 1)
            
            return (clip_tensor, sd_tensor, new_image.size)
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None

    def process_batch(self, batch_files, target_size, min_dimensions):
        """Process a batch of images"""
        clip_batch = []
        sd_batch = []
        valid_files = []
        
        for img_path in batch_files:
            result = self.process_single_image(img_path, target_size, min_dimensions)
            if result:
                clip_tensor, sd_tensor, size = result
                clip_batch.append(clip_tensor)
                sd_batch.append(sd_tensor)
                valid_files.append((img_path, size))
        
        return clip_batch, sd_batch, valid_files

    def get_text_features(self, queries):
        """Get text features for both models"""
        # CLIP text features
        tokenizer = open_clip.get_tokenizer('ViT-B-16')
        text_tokens = tokenizer(queries).cuda()
        with torch.no_grad():
            clip_text_features = self.clip_model.encode_text(text_tokens)
            clip_text_features /= clip_text_features.norm(dim=-1, keepdim=True)
        
        # SD text features
        with torch.no_grad():
            sd_text_features = []
            for query in queries:
                text_feature = self.sd_encoder.encode_text(query)
                sd_text_features.append(text_feature)
            sd_text_features = torch.stack(sd_text_features)
        
        return clip_text_features, sd_text_features

def get_location_score(image_path: Union[str, Path], query_location: str) -> float:
    """Calculate location similarity score"""
    try:
        image_gps = get_gps_data(image_path)
        if not image_gps:
            return 0.0
        
        distance = haversine_distance(image_gps, query_location)
        # Convert distance to a 0-100 score (closer = higher score)
        return max(0.0, 100.0 * (1.0 - distance / 1000.0))  # 1000km max distance
    except Exception as e:
        print(f"Error calculating location score: {e}")
        return 0.0

# Update the combine_scores method in HybridImageSearch class
def combine_scores(self, clip_scores, sd_scores, query, weight, location_scores=None):
    """Combine normalized scores from both models and location if available"""
    path_to_scores = {}
    
    # Process CLIP scores
    for result in clip_scores:
        path_to_scores[result['path']] = {
            'clip_score': result['score'],
            'sd_score': 0.0,
            'location_score': location_scores.get(result['path'], 0.0) if location_scores else 0.0
        }
    
    # Process SD scores
    for result in sd_scores:
        path = result['path']
        if path in path_to_scores:
            path_to_scores[path]['sd_score'] = result['score']
        else:
            path_to_scores[path] = {
                'clip_score': 0.0,
                'sd_score': result['score'],
                'location_score': location_scores.get(path, 0.0) if location_scores else 0.0
            }
    
    # Calculate combined scores
    results = []
    for path, scores in path_to_scores.items():
        # Base score from CLIP and SD
        combined_score = (
            self.clip_weight * scores['clip_score'] +
            self.sd_weight * scores['sd_score']
        )
        
        # Add location influence if enabled
        if location_scores:
            combined_score = combined_score * 0.7 + scores['location_score'] * 0.3
        
        combined_score *= weight
        
        results.append({
            'path': path,
            'mean_score': combined_score,
            'individual_scores': {
                f"{query} (CLIP)": scores['clip_score'],
                f"{query} (SD)": scores['sd_score'],
                f"{query} (Location)": scores['location_score'] if location_scores else 0.0
            }
        })
    
    return results

async def find_similar_images_hybrid(
    root_dir: Union[str, Path],
    weighted_queries: List[Tuple[str, float]],
    batch_size: int = 32,
    min_score: float = 20.0,
    min_dimensions: Tuple[int, int] = (500, 500),
    cache_dir: str = "search_cache",
    clip_weight: float = 0.6,
    sd_weight: float = 0.4,
    target_size: Tuple[int, int] = (512, 512),
    use_location: bool = True
):
    searcher = HybridImageSearch(clip_weight, sd_weight)
    searcher.create_indices(cache_dir)
    
    # Get location data if enabled
    location_infos = []
    if use_location:
        for query, weight in weighted_queries:
            try:
                loc_data = await get_location_from_query(query)
                if loc_data and loc_data.get('has_location', False):
                    loc_data['weight'] = weight  # Add query weight to location data
                    location_infos.append(loc_data)
                    print(f"Found location in query '{query}': {loc_data['location_name']}")
            except Exception as e:
                print(f"Warning: Location service error for query '{query}': {e}")
                continue
    
    # Process queries
    queries, weights = zip(*weighted_queries)
    clip_text_features, sd_text_features = searcher.get_text_features(queries)
    
    # Search and combine results
    all_results = []
    for query, clip_query, sd_query, weight in zip(
        queries, clip_text_features, sd_text_features, weights
    ):
        try:
            # Get raw results
            clip_results = searcher.clip_index.search(
                clip_query.cpu().numpy(), k=1000, min_score=-1.0
            ) or []
            sd_results = searcher.sd_index.search(
                sd_query.cpu().numpy(), k=1000, min_score=-1.0
            ) or []
            
            # Normalize scores
            clip_scores = searcher.normalize_scores(
                clip_results, [r['mean_score'] for r in clip_results]
            )
            sd_scores = searcher.normalize_scores(
                sd_results, [r['mean_score'] for r in sd_results]
            )
            
            # Get location scores if enabled and locations found
            location_scores = {}
            if use_location and location_infos:
                try:
                    for result in clip_scores + sd_scores:
                        path = result['path']
                        if path not in location_scores:
                            # Calculate location score using existing location data
                            location_scores[path] = calculate_location_score(
                                path, location_infos
                            )
                except Exception as e:
                    print(f"Error processing location data: {e}")
                    pass
            
            # Combine scores with location if available
            results = searcher.combine_scores(
                clip_scores, 
                sd_scores, 
                query, 
                weight,
                location_scores if use_location and location_infos else None
            )
            all_results.extend(results)
            
        except Exception as e:
            print(f"Error processing query '{query}': {str(e)}")
            continue
    
    if not all_results:
        return []
    
    # Sort and deduplicate results
    all_results.sort(key=lambda x: x['mean_score'], reverse=True)
    seen_paths = set()
    final_results = []
    
    for result in all_results:
        if result['path'] not in seen_paths and result['mean_score'] > min_score:
            seen_paths.add(result['path'])
            final_results.append(result)
    
    return final_results[:100]

def calculate_location_score(image_path: Union[str, Path], location_infos: List[dict]) -> float:
    """Calculate combined location similarity score from multiple location queries"""
    try:
        image_gps = get_gps_data(image_path)
        if not image_gps:
            return 0.0
            
        # Convert Fraction objects to float if needed
        if isinstance(image_gps, dict):
            img_lat = float(image_gps['latitude'] if isinstance(image_gps['latitude'], (int, float)) 
                          else float(image_gps['latitude'].numerator) / float(image_gps['latitude'].denominator))
            img_lon = float(image_gps['longitude'] if isinstance(image_gps['longitude'], (int, float))
                          else float(image_gps['longitude'].numerator) / float(image_gps['longitude'].denominator))
        else:
            img_lat, img_lon = float(image_gps[0]), float(image_gps[1])
        
        # Calculate weighted average of scores for all location queries
        total_weight = 0.0
        total_score = 0.0
        
        for loc_info in location_infos:
            weight = loc_info.get('weight', 1.0)
            # Get coordinates from location info
            query_lat = float(loc_info['latitude'])
            query_lon = float(loc_info['longitude'])
            
            distance = haversine_distance(img_lat, img_lon, query_lat, query_lon)
            # Convert distance to a 0-100 score (closer = higher score)
            score = max(0.0, 100.0 * (1.0 - distance / 1000.0))  # 1000km max distance
            
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
        
    except Exception as e:
        print(f"Error calculating location score for {image_path}: {e}")
        print(f"Image GPS: {image_gps}")  # Debug info
        print(f"Location info: {location_infos}")  # Debug info
        if isinstance(image_gps, dict):
            print(f"Parsed coordinates: lat={image_gps.get('latitude')}, lon={image_gps.get('longitude')}")
        return 0.0