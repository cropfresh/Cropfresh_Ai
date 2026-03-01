# Task 5: Fix & Implement Price Prediction Agent

> **Priority:** 🔴 P0 | **Phase:** 1 | **Effort:** 3–4 days  
> **Files:** `src/agents/price_prediction/agent.py` (fix + implement)  
> **Score Target:** 9/10 — Predictions within ±10% of actual mandi price

---

## 📌 Problem Statement

`src/agents/price_prediction/agent.py` has a **corrupted class name** (syntax error) and is a pure `NotImplementedError` stub. Need to fix and implement a working price prediction engine.

---

## 🔬 Research Findings

### Models for Mandi Price Prediction (India-Specific)
| Model | Strengths | Accuracy | Complexity |
|-------|-----------|----------|-----------|
| **XGBoost** | Best for multi-source features, captures non-linear patterns | MAE ~8-12% | Medium |
| **LSTM** | Excellent for time-series, captures temporal dependencies | MAE ~10-15% | High |
| **Prophet** | Handles seasonality + holidays + trend decomposition | MAE ~12-18% | Low |
| **ARIMA/SARIMA** | Classic, works well for stationary series | MAE ~15-20% | Low |
| **Ensemble** | Combined: XGBoost + LSTM + seasonal rules | MAE ~7-10% | High |

### Feature Engineering for Crop Price Prediction
```
Features:
1. Historical prices: 7d, 14d, 30d, 90d moving averages
2. Price momentum: rate of change over 7d and 30d
3. Seasonal index: month-of-year cyclic encoding (sin/cos)
4. Rainfall: district-level weekly rainfall (IMD)
5. Supply indicator: arrivals at nearby mandis (Agmarknet)
6. Festival calendar: Dussehra, Diwali, Ugadi demand spikes
7. MSP (Minimum Support Price): government floor price
8. National trend: weighted average across top 5 mandi prices
```

### Prediction Approach (Hybrid — Best for CropFresh)
1. **Rule-based baseline**: Seasonal averages + moving averages (always available)
2. **ML model**: XGBoost trained on Agmarknet historical data (when data available)
3. **LLM analysis**: Natural language reasoning about price trends (added context)

---

## 🏗️ Implementation Spec

### 1. Fix Corrupted Class Name
```python
# BEFORE (broken):
class priceprediction.Value.ToUpper()ricepriceprediction.Value.ToUpper()redictionAgent(BaseAgent):

# AFTER (fixed):
class PricePredictionAgent(BaseAgent):
```

### 2. Price Prediction Engine
```python
@dataclass
class PricePrediction:
    commodity: str
    district: str
    current_price: float
    predicted_price_7d: float
    predicted_price_30d: float
    confidence: float              # 0.0–1.0
    trend: str                     # 'rising', 'falling', 'stable'
    trend_strength: float          # 0.0–1.0 (how strong the trend is)
    seasonal_factor: str           # 'peak_harvest', 'off_season', 'normal'
    factors: list[str]             # Human-readable factors affecting prediction
    recommendation: str            # 'sell_now', 'hold_3d', 'hold_7d', 'hold_30d'
    data_source: str               # 'historical', 'model', 'llm_estimate'

class PricePredictionAgent(BaseAgent):
    """
    Price prediction using hybrid approach:
    1. Rule-based: seasonal averages + moving averages
    2. ML backbone: XGBoost (when trained model available)
    3. LLM reasoning: contextual analysis
    """
    
    async def predict(
        self,
        commodity: str,
        district: str = "Bangalore",
        days_ahead: int = 7,
    ) -> PricePrediction:
        # Fetch historical data
        history = await self._get_price_history(commodity, district, days=90)
        
        if not history:
            return await self._llm_based_prediction(commodity, district)
        
        # Calculate features
        features = self._extract_features(history)
        
        # Rule-based prediction
        rule_pred = self._rule_based_predict(features, days_ahead)
        
        # ML prediction (if model available)
        ml_pred = await self._ml_predict(features, days_ahead)
        
        # Ensemble
        if ml_pred:
            predicted = 0.6 * ml_pred + 0.4 * rule_pred
            confidence = 0.8
            source = 'model'
        else:
            predicted = rule_pred
            confidence = 0.6
            source = 'historical'
        
        # Trend analysis
        trend, strength = self._analyze_trend(history)
        
        # Seasonal context
        seasonal = self._get_seasonal_factor(commodity, datetime.now().month)
        
        # Recommendation
        recommendation = self._generate_recommendation(
            current=history[-1]['price'],
            predicted=predicted,
            trend=trend,
            seasonal=seasonal,
        )
        
        return PricePrediction(
            commodity=commodity,
            district=district,
            current_price=history[-1]['price'],
            predicted_price_7d=round(predicted, 2),
            predicted_price_30d=round(predicted * (1 + strength * 0.3), 2),
            confidence=confidence,
            trend=trend,
            trend_strength=strength,
            seasonal_factor=seasonal,
            factors=self._explain_factors(features, trend, seasonal),
            recommendation=recommendation,
            data_source=source,
        )
    
    def _rule_based_predict(self, features: dict, days_ahead: int) -> float:
        """
        Simple but effective rule-based prediction:
        predicted = 7d_avg × seasonal_multiplier × momentum_factor
        """
        avg_7d = features['avg_7d']
        seasonal_mult = features['seasonal_multiplier']
        momentum = features['momentum_7d']  # Rate of change
        
        # Weighted prediction
        predicted = avg_7d * (1 + momentum * (days_ahead / 7))
        predicted *= seasonal_mult
        
        return predicted
    
    def _analyze_trend(self, history: list[dict]) -> tuple[str, float]:
        """
        Analyze price trend from historical data.
        Returns (direction, strength).
        """
        prices = [h['price'] for h in history[-14:]]  # Last 14 days
        if len(prices) < 3:
            return 'stable', 0.0
        
        # Linear regression slope
        x = list(range(len(prices)))
        slope = np.polyfit(x, prices, 1)[0]
        avg = np.mean(prices)
        
        # Normalized slope
        rel_slope = slope / avg if avg > 0 else 0
        
        if rel_slope > 0.02:
            return 'rising', min(1.0, rel_slope * 10)
        elif rel_slope < -0.02:
            return 'falling', min(1.0, abs(rel_slope) * 10)
        else:
            return 'stable', 0.0
```

### 3. Karnataka Seasonal Calendar
```python
SEASONAL_CALENDAR = {
    'tomato': {1: 1.0, 2: 0.9, 3: 0.8, 4: 0.7, 5: 1.3, 6: 1.4, 
               7: 1.2, 8: 1.0, 9: 0.9, 10: 0.85, 11: 0.8, 12: 0.9},
    'onion':  {1: 1.0, 2: 1.1, 3: 1.2, 4: 1.3, 5: 1.1, 6: 0.9,
               7: 0.8, 8: 0.7, 9: 0.8, 10: 1.0, 11: 0.7, 12: 0.8},
    'potato': {1: 1.0, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.9, 6: 1.0,
               7: 1.1, 8: 1.2, 9: 1.1, 10: 1.0, 11: 0.95, 12: 1.0},
}
```

---

## ✅ Acceptance Criteria (9/10 Score)

| # | Criterion | Weight |
|---|-----------|--------|
| 1 | Class name fixed, no syntax errors | 10% |
| 2 | Rule-based prediction within ±15% of actual | 20% |
| 3 | Trend analysis (rising/falling/stable) correct | 15% |
| 4 | Seasonal factor for Karnataka crops included | 15% |
| 5 | Sell/hold recommendation generated | 15% |
| 6 | Graceful fallback when no historical data available | 10% |
| 7 | LLM provides natural language explanation of factors | 10% |
| 8 | Unit tests pass with mock historical data | 5% |

---

## 📚 Dependencies
- `numpy` — trend analysis, moving averages
- `src/db/postgres_client.py` → `price_history` table
- `src/scrapers/agmarknet.py` → real-time price data
