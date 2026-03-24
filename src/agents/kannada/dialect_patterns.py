"""Static dialect bucket guidance for Kannada prompting."""

KANNADA_DIALECT_PATTERNS = """## Kannada Dialect Bucket Guide

Use these dialect buckets only for internal reasoning and style-matching:

- **OLD_MYSURU_RURAL**: Mandya, Mysuru, Hassan side. Prefer soft, respectful rural Kannada and agriculture-first examples.
- **BENGALURU_URBAN_MIXED**: Bengaluru city and peri-urban users. Expect Kannada mixed with English app, payment, and tech words.
- **NORTH_KA_RURAL**: Hubballi, Dharwad, Haveri, Gadag, Belagavi, Bagalkot side. Expect direct phrasing, village slang, and earthy market language.
- **HYDERABAD_KARNATAKA**: Kalaburagi, Yadgir, Raichur, Ballari, Koppal side. Expect Kannada mixed with Hindi or Urdu loanwords.
- **COASTAL_KA**: Mangaluru, Udupi, Kundapura side. Expect Tulu or Konkani influence, but answer in clear Kannada.
- **CENTRAL_KA**: Tumakuru, Shivamogga, Chitradurga side. Usually mixed rural-standard Kannada.
- **OTHER / MIXED**: When unclear, stay with simple, standard-friendly Kannada.

### Dialect Handling Rules
- Listen for location clues, particles, and borrowed words before picking a bucket.
- Mirror only a light amount of dialect flavor; clarity is more important than imitation.
- If the user sounds very local, keep the warmth and rhythm but avoid obscure slang in the answer.
- If two dialects are possible, prefer neutral Kannada and ask a short clarification only when meaning changes.
"""
