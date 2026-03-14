"""
IMD Weather Agro Advisory Generation
"""

from .models import (
    AgroAdvisory,
    CurrentWeather,
    WeatherCondition,
    WeatherForecast,
)


def generate_agro_advisory_data(
    weather: CurrentWeather,
    forecast: WeatherForecast,
    crop: str,
) -> AgroAdvisory:
    """Generate agricultural advisory from weather data."""
    recommendations = []

    # Irrigation advisory
    if weather.rainfall_mm > 10 or weather.soil_moisture == "wet":
        irrigation = "Skip irrigation - adequate soil moisture from recent rainfall"
    elif weather.temperature_c > 35 and weather.humidity_pct < 40:
        irrigation = "Increase irrigation frequency - high evapotranspiration expected"
    else:
        irrigation = "Follow regular irrigation schedule"

    # Check upcoming rainfall
    rain_days = sum(1 for f in forecast.daily_forecasts if f.rainfall_probability_pct > 50)

    # Sowing advisory
    if rain_days >= 2:
        sowing = "Good conditions for sowing - rainfall expected"
        recommendations.append("Prepare fields for sowing before rainfall")
    elif weather.soil_moisture == "dry":
        sowing = "Delay sowing until adequate moisture available"
    else:
        sowing = "Suitable conditions for sowing if soil moisture is adequate"

    # Harvesting advisory
    if weather.condition in [WeatherCondition.HEAVY_RAIN, WeatherCondition.MODERATE_RAIN]:
        harvesting = "Delay harvesting - rain may damage harvested crop"
        recommendations.append("Cover harvested produce to protect from rain")
    elif rain_days > 0:
        harvesting = "Complete harvesting before expected rainfall"
    else:
        harvesting = "Good conditions for harvesting"

    # Pest/disease alert based on conditions
    pest_alert = ""
    if weather.humidity_pct > 80 and weather.temperature_c > 25:
        pest_alert = "High humidity - monitor for fungal diseases. Apply preventive fungicide if needed."
        recommendations.append("Scout fields for early signs of fungal infection")
    elif weather.temperature_c > 35:
        pest_alert = "High temperature stress - watch for pest outbreaks"

    # Fertilizer advisory
    if rain_days > 0:
        fertilizer = "Apply fertilizer 2-3 days before expected rainfall for better absorption"
    else:
        fertilizer = "Water plants after fertilizer application for optimal uptake"

    # Weather summary
    weather_summary = f"Current: {weather.temperature_c:.1f}°C, {weather.humidity_pct:.0f}% humidity, {weather.condition.value}. "
    if rain_days > 0:
        weather_summary += f"Rainfall expected in next {rain_days} days."
    else:
        weather_summary += "No significant rainfall expected."

    # Crop-specific advice
    if crop.lower() == "tomato":
        recommendations.append("Stake tomato plants if heavy rain expected")
        recommendations.append("Monitor for late blight in humid conditions")
    elif crop.lower() == "onion":
        recommendations.append("Reduce irrigation as crop matures")
        recommendations.append("Watch for purple blotch in humid weather")
    elif crop.lower() == "rice":
        recommendations.append("Maintain 2-3cm standing water in paddy")

    return AgroAdvisory(
        state=weather.state,
        district=weather.district,
        crop=crop,
        weather_summary=weather_summary,
        irrigation_advisory=irrigation,
        sowing_advisory=sowing,
        harvesting_advisory=harvesting,
        pest_disease_alert=pest_alert,
        fertilizer_advisory=fertilizer,
        recommendations=recommendations,
    )
