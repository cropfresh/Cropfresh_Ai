# AWS API Gateway — CropFresh AI

## Overview

Two API Gateway APIs sit between CloudFront and App Runner:

- **HTTP API** (`cropfresh-http-api`) — all REST endpoints
- **WebSocket API** (`cropfresh-ws-api`) — `/voice` persistent connection

---

## HTTP API Configuration

```yaml
# Resource: AWS::ApiGatewayV2::Api
Type: HTTP
Name: cropfresh-http-api
CorsConfiguration:
  AllowOrigins:
    - https://cropfresh.in
    - https://www.cropfresh.in
    - http://localhost:3000 # local dev buyer dashboard
  AllowMethods: [GET, POST, PUT, DELETE, OPTIONS]
  AllowHeaders: [Authorization, Content-Type, X-Request-ID]
  MaxAge: 300

DefaultRouteSettings:
  ThrottlingBurstLimit: 1000 # spike protection
  ThrottlingRateLimit: 500 # steady state req/s
```

### Route → Integration Mapping

| Route               | Method | Integration | Auth    | Throttle Override    |
| ------------------- | ------ | ----------- | ------- | -------------------- |
| `/health`           | GET    | App Runner  | ❌ None | —                    |
| `/auth/otp/request` | POST   | App Runner  | ❌ None | 10 req/s (OTP abuse) |
| `/auth/otp/verify`  | POST   | App Runner  | ❌ None | 10 req/s             |
| `/chat`             | POST   | App Runner  | ✅ JWT  | —                    |
| `/vision/analyze`   | POST   | App Runner  | ✅ JWT  | 50 req/s (heavy)     |
| `/marketplace/*`    | ANY    | App Runner  | ✅ JWT  | —                    |
| `/trade/match`      | POST   | App Runner  | ✅ JWT  | —                    |
| `/dispute/*`        | ANY    | App Runner  | ✅ JWT  | —                    |

### JWT Authorizer

```yaml
# Resource: AWS::ApiGatewayV2::Authorizer
Type: JWT
IdentitySource: $request.header.Authorization
JwtConfiguration:
  Issuer: https://cognito-idp.ap-south-1.amazonaws.com/{UserPoolId}
  # OR if using custom JWT:
  Issuer: https://api.cropfresh.in
  Audience:
    - cropfresh-mobile-app
    - cropfresh-buyer-dashboard
```

> **Note:** Farmers use SMS OTP → JWT issued by `/auth/otp/verify`. The JWT authorizer validates every subsequent request at the gateway layer — FastAPI does not re-validate (saves ~20ms per request).

---

## WebSocket API Configuration

```yaml
# Resource: AWS::ApiGatewayV2::Api
Type: WEBSOCKET
Name: cropfresh-ws-api
RouteSelectionExpression: $request.body.action

# Routes:
$connect    → Lambda Authorizer (validate JWT on connect)
$disconnect → App Runner /voice/disconnect
$default    → App Runner /voice (WebRTC signaling frames)
```

### WebSocket Flow

```
Farmer device
  │ wss://ws.cropfresh.in/voice
  ▼
API Gateway WebSocket API
  │ $connect → validate JWT (Lambda Authorizer, ~50ms)
  │ $default → proxy to App Runner /voice WebSocket
  ▼
App Runner FastAPI (/voice endpoint)
  │ WebRTC signaling + Pipecat voice pipeline
  ▼
VAD → STT → VoiceAgent → TTS → Audio stream back
```

---

## Stage Configuration

| Stage  | API Gateway URL                | Backend               |
| ------ | ------------------------------ | --------------------- |
| `dev`  | `https://api-dev.cropfresh.in` | App Runner staging    |
| `prod` | `https://api.cropfresh.in`     | App Runner production |

```yaml
# Stage variables injected to integration:
APP_RUNNER_URL:
  dev: https://cropfresh-api-staging.{region}.awsapprunner.com
  prod: https://cropfresh-api-prod.{region}.awsapprunner.com
```

---

## Custom Domain

```
api.cropfresh.in  → HTTP API (prod stage)
ws.cropfresh.in   → WebSocket API (prod stage)

TLS:  AWS ACM certificate (ap-south-1, auto-renewed)
DNS:  Route 53 A-record alias → API Gateway domain
```

---

## Cost Estimate (Phase 2–4)

| Component             | Pricing                                                   | Est. Monthly  |
| --------------------- | --------------------------------------------------------- | ------------- |
| HTTP API              | $1.00/million requests                                    | ~$2–5         |
| WebSocket API         | $1.00/million messages + $0.25/million connection-minutes | ~$3–8         |
| **Total API Gateway** |                                                           | **~$5–13/mo** |
