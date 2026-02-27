# Price Endpoints

## GET /api/prices
Get current market prices.

## GET /api/prices/predict
Get AI price prediction for a crop.

### Query Parameters
- crop_id: UUID
- mandi: string
- days_ahead: int (1-30)
