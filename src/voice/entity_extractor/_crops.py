"""
Crop name dictionaries for all 10 supported languages.

CROP_NAMES   – regional script → standard English name
COMMODITY_ALIASES – aliases / alternate spellings → standard English name
Both are merged at runtime in VoiceEntityExtractor.
"""

# ---------------------------------------------------------------------------
# Primary crop name mapping
# ---------------------------------------------------------------------------
CROP_NAMES: dict[str, str] = {
    # ── Hindi ───────────────────────────────────────────────────────────────
    "टमाटर": "tomato",       "आलू": "potato",         "प्याज": "onion",
    "गोभी": "cauliflower",   "बंद गोभी": "cabbage",    "बैंगन": "brinjal",
    "भिंडी": "okra",         "मिर्च": "chili",          "शिमला मिर्च": "capsicum",
    "गाजर": "carrot",        "मूली": "radish",          "पालक": "spinach",
    "धनिया": "coriander",    "फूलगोभी": "cauliflower",  "मटर": "peas",
    "लौकी": "bottle_gourd",  "करेला": "bitter_gourd",   "खीरा": "cucumber",
    "कद्दू": "pumpkin",      "सेब": "apple",            "केला": "banana",
    "आम": "mango",           "अंगूर": "grapes",          "संतरा": "orange",
    "अनार": "pomegranate",   "पपीता": "papaya",          "अमरूद": "guava",
    "चावल": "rice",          "गेहूं": "wheat",           "मक्का": "corn",
    "सरसों": "mustard",      "अदरक": "ginger",           "लहसुन": "garlic",

    # ── Kannada ─────────────────────────────────────────────────────────────
    "ಟೊಮೆಟೊ": "tomato",     "ಆಲೂಗಡ್ಡೆ": "potato",    "ಈರುಳ್ಳಿ": "onion",
    "ಹೂಕೋಸು": "cauliflower","ಬೀನ್ಸ್": "beans",         "ಮೆಣಸಿನಕಾಯಿ": "green_chilli",
    "ಬದನೆಕಾಯಿ": "brinjal",  "ಸೊಪ್ಪು": "spinach",       "ಕ್ಯಾರೆಟ್": "carrot",
    "ತೊಗರಿ": "pigeon_pea",  "ರಾಗಿ": "finger_millet",   "ಮೆಕ್ಕಿ": "corn",

    # ── Tamil ────────────────────────────────────────────────────────────────
    "தக்காளி": "tomato",    "உருளைக்கிழங்கு": "potato","வெங்காயம்": "onion",
    "காலிஃப்ளவர்": "cauliflower","கத்தரிக்காய்": "brinjal","வெண்டைக்காய்": "okra",
    "மிளகாய்": "chili",     "கேரட்": "carrot",          "கீரை": "spinach",
    "மாம்பழம்": "mango",    "வாழைப்பழம்": "banana",     "ஆப்பிள்": "apple",
    "நெல்": "rice",          "கோதுமை": "wheat",

    # ── Telugu ───────────────────────────────────────────────────────────────
    "టమాటా": "tomato",      "బంగాళాదుంప": "potato",   "ఉల్లిపాయ": "onion",
    "క్యాబేజీ": "cabbage",  "వంకాయ": "brinjal",        "బెండకాయ": "okra",
    "మిర్చి": "chili",       "క్యారెట్": "carrot",       "పాలక్": "spinach",
    "మామిడి": "mango",       "అరటి": "banana",           "వరి": "rice",
    "గోధుమ": "wheat",

    # ── Marathi ──────────────────────────────────────────────────────────────
    "टोमॅटो": "tomato",     "बटाटा": "potato",         "कांदा": "onion",
    "फ्लॉवर": "cauliflower","वांगे": "brinjal",          "भेंडी": "okra",
    "मिरची": "chili",
    "आंबा": "mango",         "केळ": "banana",            "तांदूळ": "rice",
    "गहू": "wheat",

    # ── Bengali ──────────────────────────────────────────────────────────────
    "টমেটো": "tomato",      "আলু": "potato",           "পেঁয়াজ": "onion",
    "ফুলকপি": "cauliflower","বেগুন": "brinjal",         "ঢেঁড়স": "okra",
    "মরিচ": "chili",         "গাজর": "carrot",           "পালং": "spinach",
    "আম": "mango",           "কলা": "banana",            "চাল": "rice",
    "গম": "wheat",

    # ── Gujarati ─────────────────────────────────────────────────────────────
    "ટામેટા": "tomato",     "બટાકા": "potato",         "ડુંગળી": "onion",
    "ફ્લાવર": "cauliflower","રીંગણ": "brinjal",          "ભીંડો": "okra",
    "મરચું": "chili",        "ગાજર": "carrot",           "પાલક": "spinach",
    "કેરી": "mango",         "કેળ": "banana",            "ચોખા": "rice",
    "ઘઉં": "wheat",

    # ── Punjabi ──────────────────────────────────────────────────────────────
    "ਟਮਾਟਰ": "tomato",     "ਆਲੂ": "potato",           "ਪਿਆਜ਼": "onion",
    "ਗੋਭੀ": "cauliflower",  "ਬੈਂਗਣ": "brinjal",         "ਭਿੰਡੀ": "okra",
    "ਮਿਰਚ": "chili",        "ਗਾਜਰ": "carrot",           "ਪਾਲਕ": "spinach",
    "ਅੰਬ": "mango",          "ਕੇਲਾ": "banana",           "ਚਾਵਲ": "rice",
    "ਕਣਕ": "wheat",

    # ── Malayalam ────────────────────────────────────────────────────────────
    "തക്കാളി": "tomato",   "ഉരുളക്കിഴങ്ങ്": "potato",  "ഉള്ളി": "onion",
    "ഗോതമ്പ്": "wheat",    "നെല്ല്": "rice",            "മുളക്": "chili",
    "വഴുതന": "brinjal",    "ഓക്ര": "okra",              "കാരറ്റ്": "carrot",
    "ആമ്പഴം": "mango",     "പഴം": "banana",

    # ── English (normalise plurals and re-entries) ───────────────────────────
    "tomato": "tomato",     "tomatoes": "tomato",
    "potato": "potato",     "potatoes": "potato",
    "onion": "onion",       "onions": "onion",
    "chilli": "chili",      "chilies": "chili",
    "brinjal": "brinjal",   "eggplant": "brinjal",
    "okra": "okra",         "ladies finger": "okra",
    "rice": "rice",         "wheat": "wheat",
    "mango": "mango",       "banana": "banana",
}

# ---------------------------------------------------------------------------
# COMMODITY_ALIASES – additional aliases / common misspellings
# Re-used as a secondary lookup merged with CROP_NAMES.
# ---------------------------------------------------------------------------
COMMODITY_ALIASES: dict[str, str] = {
    # Kannada
    'ಟೊಮೆಟೊ': 'tomato',   'ಈರುಳ್ಳಿ': 'onion',    'ಆಲೂಗಡ್ಡೆ': 'potato',
    'ಬೀನ್ಸ್': 'beans',    'ಮೆಣಸಿನಕಾಯಿ': 'green_chilli',
    # Hindi
    'टमाटर': 'tomato',     'प्याज': 'onion',       'आलू': 'potato',
    'मिर्च': 'green_chilli','गोभी': 'cauliflower',
    # Tamil
    'தக்காளி': 'tomato',   'வெங்காயம்': 'onion',  'உருளைக்கிழங்கு': 'potato',
    # Telugu
    'టమాటా': 'tomato',     'ఉల్లిపాయ': 'onion',   'బంగాళాదుంప': 'potato',
    # Marathi
    'टोमॅटो': 'tomato',    'कांदा': 'onion',       'बटाटा': 'potato',
    # Bengali
    'টমেটো': 'tomato',     'পেঁয়াজ': 'onion',     'আলু': 'potato',
    # Gujarati
    'ટામેટા': 'tomato',    'ડુંગળી': 'onion',      'બટાકા': 'potato',
    # Punjabi
    'ਟਮਾਟਰ': 'tomato',    'ਪਿਆਜ਼': 'onion',       'ਆਲੂ': 'potato',
    # Malayalam
    'തക്കാളി': 'tomato',   'ഉള്ളി': 'onion',       'ഉരുളക്കിഴങ്ങ്': 'potato',
}
