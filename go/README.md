# Go WASM Neural Network

Dieses Projekt kompiliert nach WebAssembly und stellt eine stateless API bereit.

Wichtig: Das WASM speichert keine Netzinstanzen dauerhaft.
Bei Training und Inferenz wird immer der komplette Netz-Zustand (Gewichte, Biases, Layer) als JSON uebergeben.

## Features

- Feedforward-Netz mit frei waehlbarer Schichtstruktur
- Aktivierungen: binary, logistic (sigmoid), relu
- Stateless API fuer:
  - initiales Erzeugen eines Zustands
  - Training (Backprop + SGD)
  - Forward-Pass

## Build (WASM)

```bash
cd go
GOOS=js GOARCH=wasm go build -o nn.wasm .
```

## JavaScript API

Alle Funktionen erwarten JSON-Strings und liefern JSON-Strings zurueck.

- nnCreateState(configJson)
- nnTrain(requestJson)
- nnForward(requestJson)
- nnListActivations()

### 1) Zustand erzeugen

Request fuer nnCreateState:

```json
{
  "layers": [2, 3, 1],
  "activations": ["relu", "logistic"],
  "learning_rate": 0.1
}
```

Antwort:

```json
{
  "state": {
    "layers": [2, 3, 1],
    "activations": ["relu", "logistic"],
    "learning_rate": 0.1,
    "weights": [[[...]], [[...]]],
    "biases": [[...], [...]]
  }
}
```

### 2) Trainieren

Request fuer nnTrain:

```json
{
  "state": {
    "layers": [2, 3, 1],
    "activations": ["relu", "logistic"],
    "learning_rate": 0.1,
    "weights": [
      [
        [0.1, -0.2],
        [0.3, 0.4],
        [-0.1, 0.2]
      ],
      [[0.5, -0.3, 0.1]]
    ],
    "biases": [[0, 0, 0], [0]]
  },
  "epochs": 200,
  "shuffle": true,
  "dataset": [
    { "input": [0, 0], "target": [0] },
    { "input": [0, 1], "target": [1] },
    { "input": [1, 0], "target": [1] },
    { "input": [1, 1], "target": [0] }
  ]
}
```

Antwort enthaelt den aktualisierten state und loss_history.

### 3) Inferenz

Request fuer nnForward:

```json
{
  "state": {
    "layers": [2, 3, 1],
    "activations": ["relu", "logistic"],
    "learning_rate": 0.1,
    "weights": [
      [
        [0.1, -0.2],
        [0.3, 0.4],
        [-0.1, 0.2]
      ],
      [[0.5, -0.3, 0.1]]
    ],
    "biases": [[0, 0, 0], [0]]
  },
  "input": [1, 0]
}
```

Antwort:

```json
{ "output": [0.73] }
```
