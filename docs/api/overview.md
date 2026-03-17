# API Overview - CropFresh AI

> **Last Updated:** 2026-03-17

## Base URL

- Local: `http://localhost:8000`
- Production: `https://api.cropfresh.in`

## Authentication

- API routes use `X-API-Key` outside development mode
- Shared app state provides the same ADCL, listing, and voice services to REST and voice surfaces

## Content Type

- `application/json` for REST routes
- `multipart/form-data` for voice upload routes

## Versioning

- URL prefix: `/api/v1`

## Core Surfaces

- `/api/v1/chat` for multi-agent text interactions
- `/api/v1/prices` for rate-hub queries and source health
- `/api/v1/adcl/weekly` for district-scoped weekly crop-demand intelligence
- `/api/v1/listings` and `/api/v1/orders` for marketplace flows
- `/api/v1/voice/*` and `/ws/voice/*` for voice interactions

## ADCL Weekly Report

`GET /api/v1/adcl/weekly?district=Kolar&force_live=true`

The ADCL endpoint returns:

- Report-level `week_start`, `district`, `generated_at`, `freshness`, `source_health`, and `metadata`
- Crop-level `commodity`, `green_label`, `recommendation`, `demand_trend`, `price_trend`, `seasonal_fit`, `sow_season_fit`, `buyer_count`, `total_demand_kg`, `predicted_price_per_kg`, `evidence`, `freshness`, and `source_health`
