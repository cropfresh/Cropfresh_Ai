"""Bilingual agricultural term normalization map for AgriEmbeddingWrapper."""

from __future__ import annotations

AGRI_TERM_MAP: dict[str, str] = {
        # â”€â”€ Crops (Hindi) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "tamatar": "tomato Solanum lycopersicum",
        "pyaj": "onion Allium cepa",
        "aloo": "potato Solanum tuberosum",
        "gehu": "wheat Triticum aestivum",
        "dhan": "paddy rice Oryza sativa",
        "kapas": "cotton Gossypium",
        "makka": "maize corn Zea mays",
        "til": "sesame Sesamum indicum",
        "moong": "green gram mung bean Vigna radiata",
        "urad": "black gram Vigna mungo",
        "arhar": "pigeon pea tur dal Cajanus cajan",
        "masoor": "red lentil Lens culinaris",
        "chana": "chickpea gram Cicer arietinum",
        "sarson": "mustard Brassica juncea",
        "methi": "fenugreek Trigonella foenum-graecum",
        "palak": "spinach Spinacia oleracea",
        "lauki": "bottle gourd Lagenaria siceraria",
        "karela": "bitter gourd Momordica charantia",
        "bhindi": "okra ladyfinger Abelmoschus esculentus",
        "baingan": "brinjal eggplant Solanum melongena",
        # â”€â”€ Crops (Kannada) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "togaribele": "pigeon pea tur dal Cajanus cajan",
        "hesarubele": "green gram mung bean",
        "kadale": "groundnut peanut Arachis hypogaea",
        "ragi": "finger millet Eleusine coracana",
        "jowar": "sorghum Sorghum bicolor",
        "bajra": "pearl millet Pennisetum glaucum",
        "huchellu": "sunflower Helianthus annuus",
        "shengri": "drumstick Moringa oleifera",
        "togari": "pigeon pea Karnataka tur",
        # â”€â”€ Seasons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "kharif": "kharif summer season June October rainfed monsoon crops",
        "rabi": "rabi winter season October March irrigated winter crops",
        "zaid": "zaid spring summer season March June cash crops",
        "vasant": "spring season March April planting",
        # â”€â”€ Markets & Trade â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "mandi": "APMC agricultural produce market committee wholesale market",
        "haath": "local weekly market haat bazaar periodic market",
        "bhaav": "market price commodity rate prevailing price",
        "tola": "weight unit 11.66 grams precious metals",
        "quintal": "100 kilograms bulk commodity weight",
        "fasal": "crop harvest season produce yield",
        # â”€â”€ Organizations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "kvk": "Krishi Vigyan Kendra farm science center agricultural extension",
        "fpo": "Farmer Producer Organisation collective farmer group",
        "atma": "Agricultural Technology Management Agency extension",
        "icar": "Indian Council of Agricultural Research national research institute",
        "apmc": "Agricultural Produce Market Committee regulated mandi",
        "nafed": "National Agricultural Cooperative Marketing Federation",
        # â”€â”€ Government Schemes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "pm-kisan": "PM-KISAN Pradhan Mantri Kisan Samman Nidhi income support scheme",
        "pmfby": "PMFBY Pradhan Mantri Fasal Bima Yojana crop insurance scheme",
        "kcc": "Kisan Credit Card farmer loan credit facility",
        "msp": "Minimum Support Price government guaranteed procurement price",
        "mksp": "Mahila Kisan Sashaktikaran Pariyojana women farmer empowerment",
        "pkvy": "Paramparagat Krishi Vikas Yojana organic farming scheme",
        "rkvy": "Rashtriya Krishi Vikas Yojana agricultural development scheme",
        # â”€â”€ Soil Types â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "kali mitti": "black cotton soil vertisol Deccan plateau deep irrigation",
        "ret mitti": "sandy loam soil alluvial well-drained light soil",
        "lalite mitti": "red laterite soil Karnataka acidic low fertility",
        "chiknai mitti": "clay soil heavy waterlogged paddy cultivation",
        # â”€â”€ Agricultural Inputs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "khad": "fertilizer manure nutrient NPK",
        "dawa": "pesticide insecticide fungicide agrochemical spray",
        "beej": "seed planting material variety certified",
        "sinchai": "irrigation water supply drip sprinkler flood",
        "compost": "organic compost manure bio-decomposed plant residue",
        "vermicompost": "vermicompost earthworm compost organic amendment",
        # â”€â”€ Weather â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        "barish": "rainfall precipitation rain monsoon",
        "garmi": "heat summer temperature high temperature stress",
        "sardi": "cold winter frost temperature low temperature",
        "aandhiyan": "storm cyclone wind damage crop loss",
    }


