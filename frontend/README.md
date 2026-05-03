# NN WASM Frontend (Svelte 5)

Frontend für das Go-WASM-Netz aus ../go.

## Was dieses Frontend kann

- Netzstruktur und Aktivierungen konfigurieren
- Netzwerk in Go/WASM erzeugen
- Training mit JSON-Dataset starten
- Forward-Propagation für Eingaben ausführen

## Voraussetzungen

- Node.js + npm
- Go (für go env GOROOT)
- Bereits gebautes WASM-Artefakt unter ../go/nn.wasm

## Start

```bash
cd frontend
npm install
npm run dev
```

Hinweis: Vor dev/build wird automatisch `npm run wasm:sync` ausgeführt.
Dabei werden die Dateien nach public kopiert:

- ../go/nn.wasm -> public/nn.wasm
- $GOROOT/.../wasm_exec.js -> public/wasm_exec.js

## Build

```bash
cd frontend
npm run build
```

## Relevante Dateien

- src/App.svelte: UI + Aufrufe an nnCreateNetwork/nnTrain/nnForward
- scripts/sync-wasm.mjs: Kopiert WASM-Artefakte in public/
- index.html: lädt wasm_exec.js vor dem App-Bundle
