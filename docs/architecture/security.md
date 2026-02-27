# Security — CropFresh AI

## Authentication & Authorization
- Firebase Auth with SMS OTP verification
- JWT tokens with 24-hour expiry
- Role-based access control (farmer, buyer, admin)

## Data Privacy (DPDP Act 2023 Compliance)
- Minimal data collection (phone, name, district only)
- Data stored in Indian region (GCP asia-south1)
- User consent captured before data processing
- Data deletion on user request
- No selling of farmer data to third parties

## API Security
- HTTPS everywhere (TLS 1.3)
- Rate limiting (100 req/min per user)
- Input validation (Pydantic models)
- CORS restricted to known domains
- SQL injection prevention (parameterized queries)

## Secret Management
- Environment variables for local dev
- GCP Secret Manager for production
- No secrets in code or git history
- `.env` in `.gitignore`

## Encryption
- Data at rest: Supabase PostgreSQL encryption
- Data in transit: TLS 1.3
- Sensitive fields: Phone numbers hashed for lookup

## Monitoring & Incident Response
- Automated alerts for auth failures
- Agent failure notifications
- Cost threshold alerts
- Weekly security review
