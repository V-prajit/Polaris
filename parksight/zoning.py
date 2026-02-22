import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class CoordinateHeuristicCache:
    def __init__(self, cache_filepath: str = "coord_zoning_cache.json"):
        self.cache: Dict[str, Any] = {}
        
        # Try to resolve relative to this file's directory first
        path = Path(__file__).resolve().parent.parent / cache_filepath
        if not path.exists():
            # Fallback to current working directory
            path = Path(cache_filepath)
            
        if path.exists():
            try:
                with open(path, 'r') as f:
                    self.cache = json.load(f)
                logger.debug(f"Loaded {len(self.cache)} coordinate heuristics from {path}")
            except Exception as e:
                logger.error(f"Failed to load coordinate cache at {path}: {e}")
        else:
            logger.warning(f"Coordinate cache file not found at {path}")

    def get_heuristics_by_coord(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """
        Format the incoming coordinate exactly as it was formatted in the cache.
        Returns the heuristic dictionary if found, otherwise None.
        """
        # Format the incoming coordinate exactly as it was formatted in the cache
        coord_key = f"{lat:.6f},{lon:.6f}"
        
        # Instant O(1) dictionary lookup
        data = self.cache.get(coord_key)
        
        return data

# Singleton instance for the module to use
_cache_instance = None

def get_heuristic_cache() -> CoordinateHeuristicCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CoordinateHeuristicCache("coord_zoning_cache.json")
    return _cache_instance

def estimate_parking_limits_for_coord(lat: float, lon: float) -> Optional[Dict[str, Any]]:
    """
    Takes a latitude and longitude and returns estimated parking space dimensions and 
    maximum parking structure height using a statically generated lookup table.
    """
    cache = get_heuristic_cache()
    data = cache.get_heuristics_by_coord(lat, lon)
    
    if data:
        return data
    else:
        logger.warning(f"Coordinate {lat:.6f},{lon:.6f} not found in pre-computed cache.")
        return None
