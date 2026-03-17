# Daily Rate Data Sources for Karnataka: Platforms, Coverage, and Integration Options

## Executive Summary

This report maps out the main platforms where daily rates relevant to Karnataka can be obtained, focusing on agricultural commodity prices (mandi rates), along with key consumer prices like fuel and gold where useful for farmers and traders. It groups sources into official government portals, national market platforms, state-level systems, third-party aggregators, and specialized apps, then compares them on coverage, granularity, update frequency, and integration potential for a product such as CropFresh.[^1][^2][^3][^4][^5][^6][^7][^8][^9][^10]

The universe of truly distinct, credible daily-price platforms is in the few dozens rather than hundreds; however, each exposes large numbers of markets and commodities, giving rich data once integrated. The tables below highlight where Karnataka rates can be sourced reliably and how they can be combined for user-facing comparison features.[^4][^7][^9][^11]

## 1. Core National Government Data Sources
These are the primary systems from which most other portals (including private aggregators) ultimately derive their mandi price data.

| Source | URL | Scope & Coverage | Karnataka Filter | Data Fields | Access & Integration Notes |
|--------|-----|------------------|------------------|-------------|----------------------------|
| AGMARKNET main portal | agmarknet.gov.in | National Agricultural Marketing Information Network; commodity-wise and market-wise daily wholesale prices and arrivals for 1800+ markets across India.[^9][^12] | Yes – state filter includes Karnataka.[^8][^9] | Min, max, modal price, arrivals, units, market, district, state, date.[^4][^10] | Web UI; dynamic tables; data also flows into open data API on data.gov.in; scraping or API via OGD platform required.[^4][^10] |
| OGD "Current daily price of various commodities from various markets (Mandi)" | data.gov.in resource "Current Daily Price of Various Commodities from Various Markets (Mandi)" | Open Government Data mirror of AGMARKNET daily mandi prices.[^4][^10] | Yes – state, district, market fields allow Karnataka filtering.[^10] | Wholesale max, min, modal prices per commodity, market, and date.[^10] | Dataset is dynamically generated; API endpoint must be requested via OGD platform; suitable for backend ingestion when granted.[^10] |
| e-NAM Agmarknet Price Dashboard | enam.gov.in/web/dashboard/agmarknet | e-NAM dashboard for Agmarknet-linked mandi prices.[^7] | State dropdown includes Karnataka; district and APMC filters narrow to specific markets.[^7] | Min, modal, max price, arrivals, commodity, variety, unit, date.[^7] | Interactive dashboard; underlying data is from AGMARKNET; scraping possible; official APIs may require registration.[^7] |

## 2. Karnataka State Government Systems

Karnataka has its own state-level price and market information systems layered on top of, or parallel to, national feeds.

| Source | URL | Role & Coverage | Karnataka Focus | Data & Features | Integration Notes |
|--------|-----|-----------------|-----------------|-----------------|-------------------|
| KRAMA / Krishi Marata Vahini reports | krama.karnataka.gov.in/Reports/Main_rep | Official state portal for agricultural market information; provides state-level daily report, market-wise daily report, commodity-wise daily report, periodic reports, variation reports, and latest price reports.[^1] | Entirely Karnataka-focused – covers all APMCs under the state marketing department.[^1] | Per-market and per-commodity daily prices, arrivals, state summaries.[^1] | HTML reports suitable for scraping; structure is tabular and predictable; potential for scheduled scraping or, if available, internal APIs via state government.[^1] |
| KRAMA Minimum Support / Floor Price section | krama.karnataka.gov.in/Markets/minimumsupportprice | Publishes minimum floor prices scheme for each year (e.g., 25–26) with commodity-wise support prices.[^13] | Karnataka commodities covered under state schemes.[^13] | Commodity, variety, support price (Rs/quintal). Not strictly daily but relevant reference price.[^13] | Useful for showing MSP/floor price comparisons alongside daily mandi rates; static tables easy to ingest.[^13] |
| Linked legacy portals (KSAMB, Maratavahini) | ksamb.com and maratavahini.kar.nic.in (referenced in Agmarknet documentation) | Karnataka State Agricultural Marketing Board (KSAMB) and older Krishi Marata Vahini presence referenced as state price websites.[^9] | Karnataka only.[^9] | Market profiles, charges, possibly historical price archives and GIS-based market atlas.[^9] | Treat mainly as documentation/historical reference; for live daily integration, KRAMA is the primary current portal.[^1][^9] |
| Karnataka Agricultural Price Commission (KAPRICOM) | kapricom.karnataka.gov.in (and related reports) | Price commission that analyzes costs, MSP gaps, and proposes assured price and stable market mechanisms for Karnataka crops.[^14][^15] | Karnataka-specific policy and model prices.[^14][^15] | Research reports, decision support system (KRIPA) for monitoring price crashes; recommended prices by crop.[^15] | More strategic than transactional; can be scraped for analytical overlays (e.g., profitability vs mandi price).[^15] |

## 3. National Market Platforms and Exchanges

Although CropFresh will focus on physical mandi rates, futures and reference prices from exchanges are useful benchmarks.

| Source | URL | Type | Karnataka Relevance | Data Provided | Integration Notes |
|--------|-----|------|---------------------|---------------|-------------------|
| e-NAM main portal | enam.gov.in | National Agriculture Market linking multiple APMCs; live trade and price dashboard.[^16][^7] | Covers many Karnataka mandis that are on-boarded to e-NAM.[^16] | Live traded prices, volumes, arrivals, commodity-level dashboards.[^16] | APIs are restricted to official participants; for a startup, screen-scraping dashboards is technically possible but must respect terms of use.[^16] |
| NCDEX | ncdex.com (not directly in search results but referenced via Agmarknet/docs) | Agricultural futures exchange; provides near-real-time futures prices for key agri commodities.[^9] | Futures for crops grown in Karnataka (e.g., pulses, cereals, spices) are indirectly relevant as reference prices.[^9] | Contract-wise last traded price, volume, open interest, daily OHLC.
| Combined via Agmarknet and exchange websites.[^9][^17] | Use mainly for analytics, not as farmer-facing spot price; check exchange-provided APIs or paid data vendors.[^11] |
| Agriwatch | oldwebsite.agriwatch.in | Private data service aggregating daily spot market prices over 250 commodities and 400 markets in India.[^11] | Includes many markets in South India; Karnataka markets covered as part of all-India grid.[^11] | Daily price series with 10 years of history and charting tools.[^11] | Paid subscription; not a free API; integration would require commercial agreement.[^11] |

## 4. Third-Party Web Portals Aggregating Karnataka Mandi Prices

These sites usually ingest AGMARKNET and/or state data and re-present it in user-friendly formats. They are valuable as redundancy sources and quick visual checks but less ideal as primary backends compared with official sources.

| Portal | URL | Focus & Coverage | Karnataka Support | Example Data | Notes for Integration |
|--------|-----|------------------|-------------------|-------------|----------------------|
| NaPanta | napanta.com/market-price/karnataka and specific markets like /bangalore/bangalore or /bangalore/ramanagara | Smart Kisan app and portal for farmers; shows wholesale mandi market prices as of today for each market.[^18][^19] | Dedicated Karnataka section and individual APMC-level pages (e.g., Ramanagara).[^18][^19] | For each commodity at a given market: max, min, modal (or three prices), last update date.[^19] | UI is HTML with clear tables; could be scraped as a fallback/comparison layer; primary data source is not fully documented but likely AGMARKNET/state feeds.[^18][^19] |
| Agriplus | agriplus.in/prices/all/karnataka and commodity-specific pages like /prices/arecanut-betelnut-supari/karnataka | Agricultural commodity mandi prices by state and commodity.[^20][^21] | Explicit "Commodity prices in Karnataka" section with many commodities.[^20] | Tables listing state, district, market, commodity, variety, min/max/modal price, date.[^21] | Appears to be a thin layer over AGMARKNET or similar data; HTML tables are scrappable and can serve as validation to cross-check AGMARKNET parsing.[^21] |
| CommodityMarketLive | commoditymarketlive.com/mandi-price-state/karnataka | Shows today's mandi price in Karnataka as state averages and market-wise details.[^22] | Karnataka-specific state page plus individual APMC rows.[^22] | State-average and APMC-wise min, max, modal, date per commodity in INR per quintal.[^22] | Another AGMARKNET-based aggregator; can be scraped for redundancy and as a reference check for your AGMARKNET integrations.[^22] |
| VegetableMarketPrice | vegetablemarketprice.com/market/karnataka/today | Karnataka vegetable market price today; retail and wholesale vegetable rates with history links.[^23] | Offers a Karnataka-wide vegetable list with links to specific cities/mandis.[^23] | Commodity name, price, retail price range, units, and date.[^23] | Appears retail-focused; depends on unspecified data sources; treat as supplementary reference for consumer prices rather than primary backend.[^23] |
| TodayPriceRates | market.todaypricerates.com/Karnataka-vegetables-price and city pages such as /Bangalore-vegetables-price-in-Karnataka | Live vegetable prices with daily and 7-day trend for cities in Karnataka.[^24][^25] | Karnataka-wide vegetable price page plus Bangalore-specific subpage.[^24][^25] | For each vegetable: unit price and percentage change vs previous day; separate pages for city-level retail prices.[^24][^25] | Likely uses local market surveys or mixed sources; good for consumer-facing price comparison, but provenance less transparent than AGMARKNET.[^24][^25] |
| Shyali APMC Prices | shyaliproducts.com/apmc-prices | Live daily agricultural commodity prices across India, explicitly sourced from AGMARKNET.[^26] | Supports search by state/district/market, so Karnataka mandis are included.[^26] | Category-wise prices (grains, pulses, oilseeds, fruits, vegetables, spices) linked to AGMARKNET official data.[^26] | Frontend for AGMARKNET, clearly acknowledging data origin; a convenient human-facing interface; backend integration should use AGMARKNET directly where possible.[^26] |
| General commodity portals (CommodityOnline, Economic Times Market, etc.) | Various | These provide broader commodity news and sometimes retail or wholesale prices across India.[^17] | Karnataka can be filtered via location/state for some commodities.[^17] | Spot prices, retail quotes, and futures references for staples such as rice, wheat, pulses.[^17] | Best used for news and macro trends; not suitable as high-granularity mandi backend unless official API exists.[^17] |

## 5. Mobile Apps and App-Like Portals

These are mainly farmer apps that read national or state data and push it via mobile UI and notifications.

| App / Service | Platform / URL | Data Source | Karnataka Coverage | Usage Pattern |
|---------------|----------------|------------|--------------------|--------------|
| Kisan Suvidha | Android app referenced by MANAGE and MoA&FW | Pulls mandi prices from AGMARKNET for all supported states.[^27][^28][^4] | Karnataka is one of the supported states; users can select nearby mandi and commodity.[^27] | End-user app for farmers; not a backend data source, but can be mirrored conceptually in CropFresh UX.[^27][^28] |
| Mandi Bhav India App | com.livertigo.mandibhav on Google Play | Aggregates daily mandi prices for multiple states from various data sources.[^29] | Includes southern states; though description does not name Karnataka explicitly, coverage is all-India.[^29] | End-user mobile app giving per-mandi mandi bhav for grains, vegetables, etc.[^29] |
| NaPanta Smart Kisan App | Linked to napanta.com | Uses its own collected data plus possibly AGMARKNET/state feeds to show mandi prices; over 3.3 lakh farmers as of March 2026.[^18][^19] | Strong user base in Andhra Pradesh and Karnataka with dedicated market pages.[^18][^19] | Provides push notifications/alerts for selected markets and crops; potential model for CropFresh's alert system.[^18][^19] |

## 6. Non-Agri Daily Rates Relevant to Karnataka Users

While CropFresh is agri-focused, farmers care about fuel and gold prices, which can be surfaced as value-add widgets.

| Rate Type | Portal | URL | Karnataka Support | Data Fields | Integration Considerations |
|----------|--------|-----|-------------------|------------|---------------------------|
| Petrol and diesel prices | PetrolDieselPrice.com Karnataka page | petroldieselprice.com/Karnataka-petrol-diesel-price | State-level daily retail selling prices for petrol and diesel in Karnataka.[^5] | Date, petrol price, diesel price, daily change, past revision history.[^5] | HTML tables; easily scrappable; must be treated as indicative as official OMC apps/sites are primary authorities.[^5] |
| Fuel prices (alternative) | Park+ fuel price tracker (Karnataka article) | parkplus.io/fuel-price/karnataka | Article describing dynamic fuel pricing and linking to daily updated prices by location.[^2] | Karnataka-wide coverage, with place search.[^2] | Park+ aggregates OMC data; good backup source or UX reference.[^2] |
| Gold rates | BusinessLine Gold Rate Today – Karnataka | thehindubusinessline.com/gold-rate-today/Karnataka | Daily gold prices in Karnataka for 22 kt and 24 kt; includes per-gram, 8g, 10g, and 100g with yesterday comparison.[^3] | Tables listing gold rates over recent days, with per-gram prices and price changes.[^3] | Suitable as a value-added widget; integration via scraping; underlying data powered by BankBazaar.[^3] |
| Gold rates (alternative) | IIFL Gold Rate in Karnataka | iifl.com/gold-rates-today/gold-rate-karnataka | Live gold prices for 22K and 24K with historical 10-day series.[^6] | State-level gold prices, daily series.[^6] | Another option for cross-checking gold prices; scraping needed; check terms of use.[^6] |

## 7. How Many Distinct Platforms Are Practically Useful?

Across the categories above, the practically useful platforms for Karnataka daily prices can be counted as:

- National official sources: AGMARKNET web, AGMARKNET via OGD (data.gov.in), e-NAM price dashboard (3 distinct endpoints).[^7][^8][^10][^4]
- Karnataka state systems: KRAMA reports (including daily, commodity-wise, variation, latest price), KRAMA support price section, KSAMB/legacy Marata Vahini, KAPRICOM (4–5 distinct utilities).[^13][^9][^15][^1]
- National market/exchange references: e-NAM main portal, NCDEX, Agriwatch (3 primary platforms for reference pricing).[^16][^9][^11]
- Third-party price aggregators clearly showing Karnataka content: NaPanta, Agriplus, CommodityMarketLive, VegetableMarketPrice, TodayPriceRates, Shyali APMC portal, broader commodity portals (roughly 7–8 distinct services).[^20][^18][^22][^26][^17][^23][^21][^24]
- Mobile and app-like services: Kisan Suvidha, Mandi Bhav app, NaPanta app (3 primary examples).[^18][^29][^19][^27][^28]
- Non-agri but farmer-relevant rates: 2–3 fuel price portals and 2 gold-rate portals covering Karnataka.[^2][^3][^5][^6]

This results in on the order of 20–25 distinct platforms or endpoints that can be credibly integrated or monitored for Karnataka-related daily rates, each of which opens access to hundreds of markets and commodities rather than needing 100+ distinct named platforms.[^22][^9][^11][^1][^4][^7]

## 8. Recommended Integration Strategy for CropFresh

### 8.1. Primary Backends (Authoritative)

- Use **AGMARKNET via OGD data.gov.in** and/or direct AGMARKNET scraping as the canonical source for mandi-level daily min, max, and modal prices across India, filtered to Karnataka in your ingestion pipeline.[^9][^10][^4]
- Use **KRAMA** as the authoritative state view and to capture any markets or commodities where state data is richer or timelier than AGMARKNET; reconcile both where overlaps exist.[^1]
- Use **KRAMA MSP/floor price** and **KAPRICOM reports** for reference and analytics (profitability vs support price), not as daily feeds.[^15][^13]

### 8.2. Validation and Redundancy Layers

- Regularly cross-check your parsed AGMARKNET and KRAMA data against **NaPanta**, **Agriplus**, and **CommodityMarketLive** for a sample of markets and days to detect parsing errors or missing updates.[^19][^21][^18][^22]
- Use **VegetableMarketPrice** and **TodayPriceRates** primarily to understand divergence between wholesale and retail prices, rather than as canonical wholesale sources.[^23][^25][^24]
- Where budget allows, subscribe to **Agriwatch** or similar services to get long history and analytics-grade data.[^11]

### 8.3. Value-Added Widgets

- Integrate **petrol/diesel price feeds** (e.g., through OMC apps or scrapers such as PetrolDieselPrice.com) to display daily fuel costs that affect logistics.[^5][^2]
- Show **gold rates** in Karnataka from one of the finance portals as a general economic indicator and farmer interest item.[^3][^6]

### 8.4. User-Facing Comparison Features

For end users, your UI can aggregate and compare daily prices across these layers without exposing every source name explicitly:

- Default view: canonical mandi price (AGMARKNET/KRAMA) per commodity and market.
- Comparison toggle: ability to show differences between official mandi price and one or two retail/aggregator quotes (e.g., NaPanta or VegetableMarketPrice), with a label explaining source type.
- Historical chart: use official sources for history but annotate with important MSP/floor price changes from KRAMA/KAPRICOM.

## 9. Limitations and Gaps

- There is no public list that enumerates "100+" independent Karnataka daily price platforms; most third-party sites ultimately rest on a small set of official data services.[^4][^7][^9][^11]
- Some potentially relevant APIs (e.g., OGD AGMARKNET resource) require explicit API enablement or access approval before use in production.[^10]
- Mobile apps often do not expose official APIs and would need to be treated only as UX references rather than data backends.

Within these constraints, this report has compiled the main credible sources and outlined how to combine them to provide rich, multi-layered daily rate information to users in Karnataka and beyond.[^22][^9][^11][^1][^4]

---

## References

1. [Reports](https://krama.karnataka.gov.in/Reports/Main_rep) - State Level Daily Report MarketWise Daily Report Commoditywise Daily Report Periodic Report Variatio...

2. [Check latest Petrol and Diesel Price in Karnataka here](https://parkplus.io/fuel-price/karnataka) - Are you looking to find petrol prices in Karnataka? This article will update you on the current tren...

3. [Today Gold Rate in Karnataka, 22 & 24 Carat Gold Price](https://www.thehindubusinessline.com/gold-rate-today/Karnataka/) - Gold price in Karnataka today is ₹14,745 per gram for 22 karat gold and ₹15,482 per gram for 24 cara...

4. [Current daily price of various commodities from various markets ...](https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi) - The data refers to prices of various commodities. It has the wholesale maximum price, minimum price ...

5. [Petrol Diesel price in Karnataka](http://petroldieselprice.com/Karnataka-petrol-diesel-price) - Petrol diesel price in Karnataka is Rs. 102.92 and Diesel price is Rs. 90.99 Per Litre as on 12 Feb ...

6. [Gold Rate in Karnataka - LIVE Price of 22 & 24 Carat ...](https://www.iifl.com/gold-rates-today/gold-rate-karnataka) - Today's Gold Rate in Karnataka : Compare 22 Carat & 24 Carat gold price in Karnataka. Check the hist...

7. [Agmarknet Price Dashboard - e-NAM | Trade Details](https://enam.gov.in/web/dashboard/agmarknet) - State, District, APMC's, e-NAM or Non e-NAM, Commodity, Commodity Arrivals, Commodity Variety, Price...

8. [Agmarknet Portal | National Government Services Portal](https://services.india.gov.in/service/detail/agmarknet-portal) - Agmarknet provides real-time information on market trends, commodity prices, and mandi profiles acro...

9. [[PDF] AGMARKNET - TNAU Agritech Portal](http://www.agritech.tnau.ac.in/ta/amis_ta/pdf/rs/Farmers_Centric_Portal.pdf) - The portal provides easy access to commodity-wise , variety-wise daily prices and ... Linked State W...

10. [Current Daily Price of Various Commodities from ...](https://data.gov.in/resource/current-daily-price-various-commodities-various-markets-mandi) - Current Daily Price of Various Commodities from Various Markets (Mandi) | Open Government Data (OGD)...

11. [Spot Market Prices - Agriwatch](https://oldwebsite.agriwatch.in/spot-market-prices.php) - Agriwatch collects data daily in over 250 commodities from over 400 markets in India - One of the la...

12. [AGMARKNET: Connecting farmers with markets for better prices and ...](https://www.facebook.com/kribhco.pr/posts/agmarknet-connecting-farmers-with-markets-for-better-prices-and-informed-decisio/859753182846277/) - Connecting farmers with markets for better prices and informed decisions. A one-stop portal for dail...

13. [Minimum Floor Prices Scheme for the year 25-26 - Krishimaratavahini](https://krama.karnataka.gov.in/Markets/minimumsupportprice) - Commodity Name, Variety Name, Support Price[in Rs/Quintal]. Bajra, Hybrid, 2775. Blackgram, Black Gr...

14. [Karnataka Agricultural Price Commission - ಕರ್ನಾಟಕ ಕೃಷಿ ಬೆಲೆ ಆಯೋಗ](https://kapricom.karnataka.gov.in/english) - Crop Insurance · Krishi Marata vahini · CACP. Online Services. Department Of Agriculture (KSDA) · Ag...

15. [Karnataka Agriculture Price Report 2017-18 | PDF - Scribd](https://www.scribd.com/document/712200179/Assured-Price-Stable-Market-Report-2018) - This document provides an executive summary and recommendations from the Karnataka Agriculture Price...

16. [e-NAM Mandis Trade Details - Home](https://enam.gov.in/web/dashboard/trade-data) - Mandi Board · Aspirational Districts · Dashboard · Trading Details ... Price in Rs. Commodity Arriva...

17. [Website for tracking Food commodity prices : r/mumbai - Reddit](https://www.reddit.com/r/mumbai/comments/16tjzar/website_for_tracking_food_commodity_prices/) - Agriwatch: This website provides latest information on agricultural commodities prices for major mar...

18. [Bangalore Wholesale Mandi Market prices as of Today](https://www.napanta.com/market-price/karnataka/bangalore/bangalore) - Bangalore Wholesale Mandi Market prices as of Today | 17-Mar-2026 ; Onion. Bangalore. Bangalore-Sama...

19. [Ramanagara Wholesale Mandi Market prices as of Today](https://www.napanta.com/market-price/karnataka/bangalore/ramanagara) - Ramanagara Wholesale Mandi Market prices as of Today | 17-Mar-2026 ; Carrot. Ramanagara. Carrot, ₹ 2...

20. [Agriculture Commodity Mandi prices in Karnataka - Agriplus.in](https://agriplus.in/prices/all/karnataka) - Commodity prices in Karnataka · Alasande Gram · Alsandikai · Apple · Arecanut(Betelnut/Supari) · Arh...

21. [Arecanut(Betelnut/Supari) Mandi prices in Karnataka](https://agriplus.in/prices/arecanut-betelnut-supari/karnataka) - Arecanut(Betelnut/Supari) prices in Karnataka ; Market : Sulya APMC. Variety : Cqca. Min Price 18000...

22. [Today's Mandi Price in Karnataka](https://www.commoditymarketlive.com/mandi-price-state/karnataka) - Today's Mandi Price in Karnataka - State Averages ; Ladies Finger, ₹ 25.39, ₹ 2,539.08 ; Maize, ₹ 21...

23. [Karnataka Vegetable Market Price Today, Live Market Rates](https://vegetablemarketprice.com/market/karnataka/today) - Karnataka today's Vegetable Price. |. View History data. <. March 12, 2026. > Vegetable, Price, Reta...

24. [Karnataka Vegetable Mandi Price List (Mar 2026)](https://market.todaypricerates.com/Karnataka-vegetables-price) - Vegetables price in Karnataka ; Amaranth leaves, Kg / Pcs, ₹ 17 △ 19.5%, ₹ 20 - 26 ; Amla, Kg / Pcs,...

25. [Today's Vegetable Price in Bangalore (16 Mar 2026)](https://market.todaypricerates.com/Bangalore-vegetables-price-in-Karnataka) - Vegetables price in Bangalore ; Eggplant (Brinjal or Aubergine), Kg / Pcs, ₹ 30 ▽ 3.3% ; Brinjal ( B...

26. [APMC Market Prices - Live Daily Agricultural Commodities Prices in ...](https://www.shyaliproducts.com/apmc-prices) - Explore live daily market prices of agricultural commodities including spices, grains, pulses, and o...

27. [Check prices and arrivals of agricultural commodities](https://services.india.gov.in/service/detail/check-prices-and-arrivals-of-agricultural-commodities) - Check online prices and arrivals of agricultural commodities in different markets. Users can get the...

28. [Useful agricultural web portals and mobile apps - MANAGE](https://www.manage.gov.in/fpoacademy/portalsapps.asp) - Karnataka government will soon launch a mobile app to help farmers sell their produce directly to re...

29. [Mandi Bhav App - मंडी भाव – Apps on Google Play](https://play.google.com/store/apps/details?id=com.livertigo.mandibhav&hl=en_IN) - Mandi Bhav India App is your one stop solution to every commodity market price. Get daily updated ha...

