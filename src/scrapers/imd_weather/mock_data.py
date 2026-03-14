"""
IMD Weather Mock Data Generation
"""

import random
from datetime import datetime, timedelta
from typing import Optional

from .models import (
    AlertSeverity,
    CurrentWeather,
    DailyForecast,
    WeatherAlert,
    WeatherCondition,
)


def get_mock_current_weather(state: str, district: str) -> CurrentWeather:
    """Generate mock current weather."""
    # Seasonal variation (January)
    base_temp = random.uniform(18, 28)
    humidity = random.uniform(45, 75)

    conditions = [
        WeatherCondition.SUNNY,
        WeatherCondition.PARTLY_CLOUDY,
        WeatherCondition.CLEAR,
    ]
    condition = random.choice(conditions)

    return CurrentWeather(
        state=state.title(),
        district=district.title(),
        station=f"{district.title()} Observatory",
        temperature_c=base_temp,
        feels_like_c=base_temp + random.uniform(-2, 2),
        temp_min_c=base_temp - 5,
        temp_max_c=base_temp + 8,
        humidity_pct=humidity,
        pressure_hpa=random.uniform(1008, 1018),
        rainfall_mm=0 if condition == WeatherCondition.SUNNY else random.uniform(0, 5),
        wind_speed_kmh=random.uniform(5, 20),
        wind_direction=random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
        condition=condition,
        description=condition.value.replace("_", " ").title(),
        soil_moisture=random.choice(["normal", "normal", "dry", "wet"]),
        soil_temperature_c=base_temp - 3,
        sunrise="06:45",
        sunset="18:15",
        source="mock",
    )


def get_mock_forecast_data(
    state: str,
    district: str,
    days: int,
) -> list[DailyForecast]:
    """Generate mock forecast data."""
    forecasts = []
    base_temp = random.uniform(20, 30)

    for i in range(days):
        date = datetime.now() + timedelta(days=i)
        temp_variation = random.uniform(-3, 3)

        # Occasional rain
        rain_prob = random.uniform(0, 100)
        if rain_prob > 70:
            condition = random.choice([
                WeatherCondition.LIGHT_RAIN,
                WeatherCondition.MODERATE_RAIN,
            ])
            rainfall = random.uniform(5, 30)
        else:
            condition = random.choice([
                WeatherCondition.SUNNY,
                WeatherCondition.PARTLY_CLOUDY,
                WeatherCondition.CLEAR,
            ])
            rainfall = 0

        forecasts.append(DailyForecast(
            date=date,
            temp_max_c=base_temp + temp_variation + 5,
            temp_min_c=base_temp + temp_variation - 5,
            rainfall_probability_pct=rain_prob if rain_prob > 40 else 0,
            expected_rainfall_mm=rainfall,
            condition=condition,
            description=condition.value.replace("_", " ").title(),
            wind_speed_kmh=random.uniform(8, 25),
            humidity_pct=random.uniform(50, 80),
        ))

    return forecasts


def get_mock_alerts_data(
    state: str,
    district: Optional[str],
) -> list[WeatherAlert]:
    """Generate mock weather alerts."""
    # 70% chance of no alerts
    if random.random() > 0.3:
        return []

    alert_types = [
        ("Heavy Rainfall", "Heavy rainfall expected. May cause waterlogging.", "Delay field operations. Ensure proper drainage."),
        ("Heat Wave", "Temperature expected to exceed 40°C.", "Provide shade for livestock. Increase irrigation."),
        ("Strong Winds", "Gusty winds expected up to 50 km/h.", "Secure loose materials. Support tall crops."),
    ]

    alert_type, description, farming_advisory = random.choice(alert_types)

    return [
        WeatherAlert(
            state=state.title(),
            district=district.title() if district else "All Districts",
            severity=AlertSeverity.YELLOW,
            alert_type=alert_type,
            headline=f"{alert_type} Warning",
            description=description,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=24),
            farming_advisory=farming_advisory,
        )
    ]
