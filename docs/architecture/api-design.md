# API Design — CropFresh AI

## Base URL
- **Local**: `http://localhost:8000/api`
- **Staging**: `https://cropfresh-staging-xxxxx.run.app/api`
- **Production**: `https://api.cropfresh.in/api`

## API Versioning
- URL-prefix based: `/api/v1/...`
- Breaking changes → bump version

## Authentication
- **Method**: Firebase Auth + JWT
- **Flow**: OTP via SMS → Firebase verify → JWT token
- **Header**: `Authorization: Bearer <token>`

## Response Format
```json
{
  "success": true,
  "data": { ... },
  "message": "Operation completed",
  "metadata": {
    "page": 1,
    "per_page": 20,
    "total": 100
  }
}
```

## Error Format
```json
{
  "success": false,
  "error": {
    "code": "CROP_NOT_FOUND",
    "message": "Crop with ID xxx not found",
    "details": { ... }
  }
}
```

## Rate Limiting
- **Default**: 100 requests/minute per user
- **Agent triggers**: 10 requests/minute per user
- **Public**: 30 requests/minute per IP

## Conventions
- Use snake_case for JSON fields
- Use ISO 8601 for timestamps
- Pagination via `?page=1&per_page=20`
- Filter via query params: `?crop_type=tomato&grade=A`
