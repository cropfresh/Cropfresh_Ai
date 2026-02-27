# Auth Endpoints

## POST /api/auth/otp
Send OTP to phone number.

### Request
```json
{"phone": "+919876543210"}
```n
## POST /api/auth/verify
Verify OTP and get JWT token.

### Request
```json
{"phone": "+919876543210", "otp": "123456"}
```
