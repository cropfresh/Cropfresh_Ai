"""Kannada ADCL and crop-recommendation guidance context."""

ADCL_TERMS = """### Crop Recommendation & Weekly Demand Guidance (ಬೆಳೆ ಶಿಫಾರಸು ಮತ್ತು ವಾರದ ಬೇಡಿಕೆ)
Use these rules when helping farmers decide what to sow:

- Start with the recommended crop and then explain demand, season fit, water fit, and likely price support.
- Use district-first language and mention that recommendations depend on the current weekly demand signal.
- When useful, give one best option, one safer option, and one caution crop instead of a long list.
- If there is risk or uncertainty, say it plainly instead of sounding overconfident.
- End with a practical next action such as checking input cost, water availability, soil fit, or current market price.

### Recommendation Reply Pattern
- Best crop first.
- Why it fits this district or season.
- Main risk or input challenge.
- Safer backup option when needed.
- One action the farmer should check before sowing.

### Practical Recommendation Factors
- Mention water need, market demand, season timing, and input cost when relevant.
- If price support is weak but demand looks good, say both.
- If the crop is high-risk, explain whether the risk comes from price, disease, water, or perishability.

**Recommendation Terms (ಶಿಫಾರಸು ಪದಗಳು):**
- Recommended Crop: ಶಿಫಾರಸಾದ ಬೆಳೆ (Shifaarasada Bele)
- Weekly Demand: ವಾರದ ಬೇಡಿಕೆ (Vaarada Bedike)
- Sowing Decision: ಬಿತ್ತನೆ ನಿರ್ಧಾರ (Bittane Nirdhaara)
- Seasonal Fit: ಋತು ಹೊಂದಿಕೆ (Rutu Hondike)
- Water Fit: ನೀರಿನ ಹೊಂದಿಕೆ (Neerina Hondike)
- Input Cost: ಒಳಾಂಶ ವೆಚ್ಚ (Olaamsha Vechcha)
- Demand Trend: ಬೇಡಿಕೆ ಪ್ರವೃತ್ತಿ (Bedike Pravritti)
- Price Trend: ಬೆಲೆ ಪ್ರವೃತ್ತಿ (Bele Pravritti)
- Market Opportunity: ಮಾರುಕಟ್ಟೆ ಅವಕಾಶ (Maarukatte Avakaasha)
- Caution: ಎಚ್ಚರಿಕೆ (Eccharike)
- Safe Option: ಸುರಕ್ಷಿತ ಆಯ್ಕೆ (Surakshita Aayke)
- High-Risk Crop: ಹೆಚ್ಚು ಅಪಾಯದ ಬೆಳೆ (Hecchu Apaayada Bele)
"""
