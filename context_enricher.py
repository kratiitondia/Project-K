from typing import Dict, Any, Optional

EXAMPLE_NEARBY = [
    {"name": "Starbucks", "distance_m": 50, "offers": ["10% Hot Cocoa"]},
    {"name": "Local Clothing Store", "distance_m": 200, "offers": ["Flat 15% on winter wear"]}
]

class ContextEnricher:
    @staticmethod
    def enrich(user_id: str, lat: Optional[float], lon: Optional[float]) -> Dict[str, Any]:
        # Replace with real API calls in production
        return {
            "user_id": user_id,
            "location": {"lat": lat, "lon": lon},
            "nearby_stores": EXAMPLE_NEARBY,
            "user_profile": {"loyalty_tier": "gold", "coupons": ["10% Hot Cocoa"]}
        }
