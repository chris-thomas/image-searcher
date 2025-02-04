import torch
import open_clip
from PIL import Image
from pathlib import Path
import numpy as np
import json
import hashlib
from typing import List, Tuple, Union
from PIL.ExifTags import TAGS, GPSTAGS
from math import radians, sin, cos, sqrt, atan2

class FeatureCache:
    def __init__(self, cache_dir="cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.index_file = self.cache_dir / "cache_index.json"
        self.load_index()
    
    def load_index(self):
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}
            
    def save_index(self):
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f)
    
    def get_cache_key(self, path):
        mtime = Path(path).stat().st_mtime
        return f"{path}_{mtime}"
    
    def get_features(self, image_path):
        key = self.get_cache_key(image_path)
        if key in self.index:
            cache_path = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.npy"
            if cache_path.exists():
                return np.load(str(cache_path))
        return None
    
    def save_features(self, image_path, features):
        key = self.get_cache_key(image_path)
        cache_path = self.cache_dir / f"{hashlib.md5(key.encode()).hexdigest()}.npy"
        np.save(str(cache_path), features.cpu().numpy())
        self.index[key] = str(cache_path)
        self.save_index()

_model = None
_preprocess = None

def get_model():
    """Get or initialize the CLIP model"""
    global _model, _preprocess
    if _model is None:
        _model, _, _preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
        _model = _model.cuda()
    return _model, _preprocess

def get_gps_data(image_path):
    """Extract GPS data from image EXIF"""
    try:
        image = Image.open(image_path)
        exif = image._getexif()
        if not exif:
            return None
        
        for tag_id in exif:
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                gps_data = {}
                for gps_tag in exif[tag_id]:
                    sub_tag = GPSTAGS.get(gps_tag, gps_tag)
                    gps_data[sub_tag] = exif[tag_id][gps_tag]
                
                if 'GPSLatitude' in gps_data and 'GPSLongitude' in gps_data:
                    lat = gps_data['GPSLatitude']
                    lon = gps_data['GPSLongitude']
                    lat_ref = gps_data.get('GPSLatitudeRef', 'N')
                    lon_ref = gps_data.get('GPSLongitudeRef', 'E')
                    
                    lat = sum(x/y for x, y in zip(lat, (1, 60, 3600)))
                    lon = sum(x/y for x, y in zip(lon, (1, 60, 3600)))
                    
                    if lat_ref == 'S': lat = -lat
                    if lon_ref == 'W': lon = -lon
                    
                    return {'latitude': lat, 'longitude': lon}
    except:
        pass
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers"""
    R = 6371  # Earth's radius in kilometers
    
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c
