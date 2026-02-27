// Seed graph data
CREATE (tomato:Crop {name: 'Tomato', name_kn: 'ಟೊಮ್ಯಾಟೊ'})
CREATE (hubli:Mandi {name: 'Hubli APMC', district: 'Dharwad'})
CREATE (tomato)-[:TRADED_AT]->(hubli)
