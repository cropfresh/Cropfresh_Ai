# AWS VPC and Networking - CropFresh AI

Region: `ap-south-1` (Mumbai)

---

## VPC Layout

```text
VPC: cropfresh-vpc
CIDR: 10.0.0.0/16

Availability Zones: ap-south-1a, ap-south-1b

+------------------------------------------------------------------+
| cropfresh-vpc (10.0.0.0/16)                                      |
|                                                                  |
|  Public Subnet 1a          Public Subnet 1b                      |
|  10.0.1.0/24               10.0.2.0/24                           |
|  NAT Gateway               NAT standby                           |
|                                                                  |
|  Private Subnet 1a         Private Subnet 1b                     |
|  10.0.11.0/24              10.0.12.0/24                          |
|  RDS PostgreSQL            RDS standby / future private services |
|                                                                  |
|  App Runner VPC Connector -> attached to private subnets         |
+------------------------------------------------------------------+
```

Internet Gateway -> Public subnets
NAT Gateway -> Private subnets outbound for Groq, Together, Qdrant, Neo4j, Redis, and other external APIs

If a private vLLM or GPU inference service is deployed inside AWS, keep that traffic inside the VPC instead of sending it through NAT.

---

## Security Groups

### `cropfresh-rds-sg`

Attached to: AWS RDS PostgreSQL

| Direction | Protocol | Port | Source | Purpose |
|-----------|----------|------|--------|---------|
| Inbound | TCP | 5432 | `cropfresh-apprunner-sg` | App Runner -> RDS |
| Outbound | All | All | - | Default return traffic |

### `cropfresh-apprunner-sg`

Attached to: App Runner VPC Connector

| Direction | Protocol | Port | Source | Purpose |
|-----------|----------|------|--------|---------|
| Outbound | TCP | 5432 | `cropfresh-rds-sg` | App Runner -> PostgreSQL |
| Outbound | TCP | 443 | `0.0.0.0/0` | Groq, Together, Qdrant, Neo4j, Redis, general HTTPS egress |
| Outbound | TCP | 13641 | `0.0.0.0/0` | Redis Cloud, if used |

---

## App Runner VPC Connector

```bash
aws apprunner create-vpc-connector \
  --vpc-connector-name cropfresh-vpc-connector \
  --subnets subnet-PRIVATE-1a subnet-PRIVATE-1b \
  --security-groups sg-APPRUNNER-ID
```

The VPC connector gives App Runner access to:

- Private PostgreSQL in the VPC
- Internal AWS services through VPC networking
- Outbound internet through NAT for external model and data providers

---

## RDS PostgreSQL Subnet Group

```bash
aws rds create-db-subnet-group \
  --db-subnet-group-name cropfresh-rds-subnet-group \
  --db-subnet-group-description "CropFresh RDS private subnets" \
  --subnet-ids subnet-PRIVATE-1a subnet-PRIVATE-1b
```

---

## Route 53 DNS Entries

| Record | Type | Value | TTL |
|--------|------|-------|-----|
| `api.cropfresh.in` | A (Alias) | API custom domain | - |
| `ws.cropfresh.in` | A (Alias) | Websocket custom domain | - |
| `cropfresh.in` | A (Alias) | Web frontend or CloudFront | - |
| `www.cropfresh.in` | CNAME | `cropfresh.in` | 300 |

---

## Cost Estimate

| Component | Estimated Monthly Cost |
|-----------|------------------------|
| NAT Gateway (Phase 2-4) | ~`$35` |
| App Runner VPC Connector | Included in App Runner |
| Route 53 Hosted Zone | ~`$0.50` |
| **Total networking** | **~`$35/mo`** |

---

## Optimization Note

NAT Gateway remains the biggest networking cost driver. Prefer:

- VPC endpoints for AWS-native services such as S3 and Secrets Manager
- Private connectivity for any self-hosted vLLM or GPU inference service
- Only using outbound NAT for providers that truly require public HTTPS egress
