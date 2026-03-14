"""
IMD Weather API Constants
"""

# API Endpoints
IMD_API_BASE = "https://api.imd.gov.in/v1"
OWM_API_BASE = "https://api.openweathermap.org/data/2.5"

# District coordinates (for OpenWeatherMap fallback)
DISTRICT_COORDS = {
    ("karnataka", "kolar"): (13.1333, 78.1333),
    ("karnataka", "bangalore"): (12.9716, 77.5946),
    ("karnataka", "mysore"): (12.2958, 76.6394),
    ("karnataka", "hubli"): (15.3647, 75.1240),
    ("karnataka", "belgaum"): (15.8497, 74.4977),
    ("maharashtra", "nashik"): (19.9975, 73.7898),
    ("maharashtra", "pune"): (18.5204, 73.8567),
    ("maharashtra", "mumbai"): (19.0760, 72.8777),
    ("tamil nadu", "chennai"): (13.0827, 80.2707),
    ("tamil nadu", "coimbatore"): (11.0168, 76.9558),
    ("andhra pradesh", "vijayawada"): (16.5062, 80.6480),
    ("telangana", "hyderabad"): (17.3850, 78.4867),
    ("gujarat", "ahmedabad"): (23.0225, 72.5714),
    ("rajasthan", "jaipur"): (26.9124, 75.7873),
    ("uttar pradesh", "lucknow"): (26.8467, 80.9462),
}
