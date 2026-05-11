# nn

Interactive Neural Network

Webseite verfügbar unter: https://tools.info-bw.de/nn

## Entwickeln

Benötigt: Node (am besten Version 24+)  
Einmalig zu Beginn: `cd frontend && npm install`

1. Go-Modul bauen:
   `cd go && GOOS=js GOARCH=wasm go build -o nn.wasm`

2. Frontend starten:
   `cd frontend && npm run dev`

## Bauen

`./build.sh`
