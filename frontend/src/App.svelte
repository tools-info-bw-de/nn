<script>
  import { onMount } from "svelte";
  import FehlerwertChart from "./lib/FehlerwertChart.svelte";
  import NetworkGraph from "./lib/NetworkGraph.svelte";
  import TrainingsModal from "./lib/TrainingsModal.svelte";
  import SevenSegment from "./lib/SevenSegment.svelte";

  const defaultDataset = JSON.stringify(
    [
      { input: [0, 0], target: [0] },
      { input: [0, 1], target: [1] },
      { input: [1, 0], target: [1] },
      { input: [1, 1], target: [0] },
    ],
    null,
    2,
  );

  const defaultInputValues = [1, 0];
  const publicAsset = (fileName) => `${import.meta.env.BASE_URL}${fileName}`;

  let wasmReady = false;
  let busy = $state(false);
  let activationMenuOpen = $state(false);
  let infoMenuOpen = $state(false);

  let availableActivations = $state(["binary", "logistic", "relu"]);

  let tabCounter = 1;
  let tabs = $state([createTab(tabCounter)]);
  // svelte-ignore state_referenced_locally
  let activeTabId = $state(tabs[0].id);

  let renamingTabId = $state("");
  let renameDraft = $state("");

  let trainingWindowPosition = $state({ x: 250, y: 150 });
  let trainingWindowSize = $state({ width: 960, height: 520 });
  let trainingWindowDragging = $state(false);
  let trainingWindowDragOffset = { x: 0, y: 0 };
  let trainingWindowResizing = $state(false);
  let trainingWindowResizeDir = "";
  let trainingWindowResizeStart = {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    left: 0,
    top: 0,
  };

  let datasetModalOpen = $state(false);
  let datasetImportPromptOpen = $state(false);
  let datasetImportStep = $state("");
  let pendingImportCsvText = "";
  let pendingImportFirstLine = $state("");
  let pendingImportSecondLine = $state("");
  let pendingImportInputCount = 0;
  let pendingImportOutputCount = 0;
  let pendingImportDelimiter = ",";

  let isTraining = $state(false);
  let stopTrainingRequested = false;
  let trainingTabId = "";
  let trainingTrainerId = "";
  let trainingEpochOffset = 0;
  let trainingLossHistoryBase = [];
  let trainingLastLoss = null;
  let trainingDeviation = null;
  let highlightedConnectionId = $state("");
  let liveInferenceRunId = 0;
  let nnWorker = null;
  let nnWorkerReadyPromise = null;
  let nnWorkerRequestId = 0;
  const nnWorkerPending = new Map();
  let networkImportInputEl = null;

  function resetWorkerInstance() {
    if (nnWorker) {
      nnWorker.onmessage = null;
      nnWorker.onerror = null;
      nnWorker.onmessageerror = null;
      nnWorker.terminate();
      nnWorker = null;
    }
    nnWorkerReadyPromise = null;
  }

  function initWorker() {
    if (nnWorkerReadyPromise) {
      return nnWorkerReadyPromise;
    }

    nnWorkerReadyPromise = new Promise((resolve, reject) => {
      const worker = new Worker(new URL("./lib/nnWorker.js", import.meta.url), {
        type: "classic",
      });
      nnWorker = worker;

      const initRequestId = ++nnWorkerRequestId;

      worker.onmessage = (event) => {
        const msg = event.data;
        if (!msg || typeof msg !== "object") {
          return;
        }

        if (msg.type === "ready" && msg.id === initRequestId) {
          resolve();
          return;
        }

        if (msg.type === "response" && msg.id === initRequestId && !msg.ok) {
          reject(
            new Error(
              msg.error || "WASM-Worker Initialisierung fehlgeschlagen.",
            ),
          );
          return;
        }

        if (msg.type !== "response") {
          return;
        }

        const pending = nnWorkerPending.get(msg.id);
        if (!pending) {
          return;
        }
        nnWorkerPending.delete(msg.id);

        if (msg.ok) {
          pending.resolve(msg.result);
        } else {
          pending.reject(new Error(msg.error || "WASM-Worker Fehler"));
        }
      };

      worker.onerror = (event) => {
        const details = [event.filename, event.lineno, event.colno]
          .filter(
            (value) => value !== undefined && value !== null && value !== "",
          )
          .join(":");
        const suffix = details ? ` (${details})` : "";
        reject(
          new Error(`${event.message || "WASM-Worker abgestürzt."}${suffix}`),
        );
      };

      worker.onmessageerror = () => {
        reject(new Error("WASM-Worker Nachricht konnte nicht gelesen werden."));
      };

      worker.postMessage({
        type: "init",
        id: initRequestId,
        baseUrl: import.meta.env.BASE_URL,
      });
    }).catch((error) => {
      resetWorkerInstance();
      throw error;
    });

    return nnWorkerReadyPromise;
  }

  async function callWorker(method, payload = {}) {
    await initWorker();

    if (!nnWorker) {
      throw new Error("WASM-Worker nicht verfügbar.");
    }

    const requestId = ++nnWorkerRequestId;
    const safePayload = clone(payload);
    return new Promise((resolve, reject) => {
      nnWorkerPending.set(requestId, { resolve, reject });
      nnWorker.postMessage({
        type: "call",
        id: requestId,
        method,
        payload: safePayload,
      });
    });
  }

  function disposeWorker() {
    for (const pending of nnWorkerPending.values()) {
      pending.reject(new Error("WASM-Worker wurde beendet."));
    }
    nnWorkerPending.clear();

    resetWorkerInstance();
  }

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function createTab(nr) {
    const layers = [2, 3, 1];
    const defaultRows = JSON.parse(defaultDataset);
    return {
      id: `tab-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      name: `Netz ${nr}`,
      layers,
      activation: "logistic",
      learningRate: 0.1,
      epochs: 0,
      showOutputSegment: false,
      shuffle: true,
      datasetRows: defaultRows,
      inputNeuronValues: Array.from({ length: layers[0] }, (_, idx) =>
        String(defaultInputValues[idx] ?? 0),
      ),
      inputNeuronNames: Array.from(
        { length: layers[0] },
        (_, idx) => `input${idx + 1}`,
      ),
      outputNeuronValues: Array.from(
        { length: layers[layers.length - 1] },
        () => "-",
      ),
      outputNeuronNames: Array.from(
        { length: layers[layers.length - 1] },
        (_, idx) => `output${idx + 1}`,
      ),
      lossHistory: [],
      trainerId: "",
      state: null,
    };
  }

  function getActiveTab() {
    const tab = tabs.find((item) => item.id === activeTabId);
    if (!tab) {
      throw new Error("Kein aktiver Tab gefunden.");
    }
    return tab;
  }

  function updateTab(tabId, updater) {
    tabs = tabs.map((tab) => {
      if (tab.id !== tabId) {
        return tab;
      }
      const next = clone(tab);
      updater(next);
      return next;
    });
  }

  function updateActiveTab(updater) {
    updateTab(activeTabId, updater);
  }

  function normalizeTabNeuronIo(tab) {
    const inputCount = tab.layers[0] ?? 0;
    const outputCount = tab.layers[tab.layers.length - 1] ?? 0;

    const prevInputs = Array.isArray(tab.inputNeuronValues)
      ? tab.inputNeuronValues
      : [];

    tab.inputNeuronValues = Array.from(
      { length: inputCount },
      (_, idx) => prevInputs[idx] ?? "0",
    );

    const prevInputNames = Array.isArray(tab.inputNeuronNames)
      ? tab.inputNeuronNames
      : [];

    tab.inputNeuronNames = Array.from({ length: inputCount }, (_, idx) => {
      const raw = String(prevInputNames[idx] ?? "").trim();
      return raw.length > 0 ? raw : `input${idx + 1}`;
    });

    const prevOutputs = Array.isArray(tab.outputNeuronValues)
      ? tab.outputNeuronValues
      : [];

    tab.outputNeuronValues = Array.from(
      { length: outputCount },
      (_, idx) => prevOutputs[idx] ?? "-",
    );

    const prevOutputNames = Array.isArray(tab.outputNeuronNames)
      ? tab.outputNeuronNames
      : [];

    tab.outputNeuronNames = Array.from({ length: outputCount }, (_, idx) => {
      const raw = String(prevOutputNames[idx] ?? "").trim();
      return raw.length > 0 ? raw : `output${idx + 1}`;
    });
  }

  function parseNeuronInputs(tab) {
    return (tab.inputNeuronValues || []).map((value) => {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : 0;
    });
  }

  function mapOutputsToStrings(tab, outputValues) {
    const outputCount = tab.layers[tab.layers.length - 1] ?? 0;
    return Array.from({ length: outputCount }, (_, idx) => {
      const raw = outputValues?.[idx];
      const parsed = Number(raw);
      return Number.isFinite(parsed) ? parsed.toFixed(4) : "-";
    });
  }

  function normalizeDatasetRows(tab) {
    const inputCount = tab.layers[0] ?? 0;
    const outputCount = tab.layers[tab.layers.length - 1] ?? 0;
    const existing = Array.isArray(tab.datasetRows) ? tab.datasetRows : [];
    const source = existing.length > 0 ? existing : [{ input: [], target: [] }];

    tab.datasetRows = source.map((row) => {
      const input = Array.from({ length: inputCount }, (_, idx) => {
        const value = Number(row?.input?.[idx] ?? 0);
        return Number.isFinite(value) ? value : 0;
      });

      const target = Array.from({ length: outputCount }, (_, idx) => {
        const value = Number(row?.target?.[idx] ?? 0);
        return Number.isFinite(value) ? value : 0;
      });

      return { input, target };
    });
  }

  function cloneDatasetRows(rows) {
    return rows.map((row) => ({
      input: [...row.input],
      target: [...row.target],
    }));
  }

  function currentDatasetRows(tab) {
    const rows = Array.isArray(tab.datasetRows) ? tab.datasetRows : [];
    return cloneDatasetRows(rows);
  }

  function detectCsvDelimiter(lines) {
    const first = lines[0] || "";
    const semicolons = (first.match(/;/g) || []).length;
    const commas = (first.match(/,/g) || []).length;
    return semicolons > commas ? ";" : ",";
  }

  function splitCsvLine(line, delimiter) {
    return line.split(delimiter).map((part) => part.trim());
  }

  let trainingImportError = $state("");
  function parseNodeCountLine(line) {
    const match = String(line || "").match(
      /^in\s*:\s*(\d+)\s*,\s*out\s*:\s*(\d+)$/i,
    );

    if (!match) {
      console.log("aah");
      trainingImportError =
        "CSV-Import abgebrochen: Erste Zeile muss im Format <code>in:x,out:y</code> vorliegen (z. B. <code>in:3,out:2</code> -> 3 Input- und 2 Output-Neuronen).";
      return;
    }

    const inputCount = Number(match[1]);
    const outputCount = Number(match[2]);

    if (
      !Number.isInteger(inputCount) ||
      !Number.isInteger(outputCount) ||
      inputCount < 1 ||
      outputCount < 1
    ) {
      throw new Error(
        "CSV-Import abgebrochen: in:x,out:y muss positive Ganzzahlen enthalten.",
      );
    }

    return { inputCount, outputCount };
  }

  function parseImportRows(
    lines,
    delimiter,
    startIndex,
    inputCount,
    outputCount,
  ) {
    const neededColumns = inputCount + outputCount;
    const dataRows = lines.slice(startIndex);

    if (dataRows.length === 0) {
      throw new Error("Keine Datenzeilen in CSV gefunden.");
    }

    return dataRows.map((line, rowIndex) => {
      const row = splitCsvLine(line, delimiter);
      if (row.length < neededColumns) {
        throw new Error(
          `CSV-Zeile ${rowIndex + 1} hat zu wenige Spalten (erwartet ${neededColumns}).`,
        );
      }

      const input = Array.from({ length: inputCount }, (_, idx) => {
        const value = Number(row[idx]);
        if (!Number.isFinite(value)) {
          throw new Error(
            `Ungültiger Input-Wert in CSV-Zeile ${rowIndex + 1}.`,
          );
        }
        return value;
      });

      const target = Array.from({ length: outputCount }, (_, idx) => {
        const value = Number(row[inputCount + idx]);
        if (!Number.isFinite(value)) {
          throw new Error(
            `Ungültiger Output-Wert in CSV-Zeile ${rowIndex + 1}.`,
          );
        }
        return value;
      });

      return { input, target };
    });
  }

  function parseImportNames(lines, delimiter, inputCount, outputCount) {
    const neededColumns = inputCount + outputCount;
    if (lines.length < 2) {
      throw new Error("Es gibt keine zweite Zeile für Knotennamen.");
    }

    const values = splitCsvLine(lines[1], delimiter);
    if (values.length < neededColumns) {
      throw new Error(
        `Die zweite Zeile hat zu wenige Knotennamen (erwartet ${neededColumns}).`,
      );
    }

    const inputNames = Array.from({ length: inputCount }, (_, idx) => {
      const raw = String(values[idx] ?? "").trim();
      return raw.length > 0 ? raw : `input${idx + 1}`;
    });

    const outputNames = Array.from({ length: outputCount }, (_, idx) => {
      const raw = String(values[inputCount + idx] ?? "").trim();
      return raw.length > 0 ? raw : `output${idx + 1}`;
    });

    return { inputNames, outputNames };
  }

  function arraysEqual(a, b) {
    if (a.length !== b.length) {
      return false;
    }
    return a.every((value, idx) => String(value) === String(b[idx]));
  }

  function closeDatasetImportPrompt() {
    datasetImportPromptOpen = false;
    datasetImportStep = "";
    pendingImportCsvText = "";
    pendingImportFirstLine = "";
    pendingImportSecondLine = "";
    pendingImportInputCount = 0;
    pendingImportOutputCount = 0;
    pendingImportDelimiter = ",";
  }

  function applyImportedDataset(
    parsedRows,
    importedNames = null,
    adoptNames = false,
  ) {
    const inputCount = pendingImportInputCount;
    const outputCount = pendingImportOutputCount;

    updateActiveTab((tab) => {
      const previousInputNames = Array.isArray(tab.inputNeuronNames)
        ? [...tab.inputNeuronNames]
        : [];
      const previousOutputNames = Array.isArray(tab.outputNeuronNames)
        ? [...tab.outputNeuronNames]
        : [];

      tab.layers[0] = inputCount;
      tab.layers[tab.layers.length - 1] = outputCount;
      tab.state = null;
      tab.lossHistory = [];

      normalizeTabNeuronIo(tab);

      tab.datasetRows = parsedRows;

      if (adoptNames && importedNames) {
        tab.inputNeuronNames = [...importedNames.inputNames];
        tab.outputNeuronNames = [...importedNames.outputNames];
      } else {
        tab.inputNeuronNames = Array.from({ length: inputCount }, (_, idx) => {
          const raw = String(previousInputNames[idx] ?? "").trim();
          return raw.length > 0 ? raw : `input${idx + 1}`;
        });

        tab.outputNeuronNames = Array.from(
          { length: outputCount },
          (_, idx) => {
            const raw = String(previousOutputNames[idx] ?? "").trim();
            return raw.length > 0 ? raw : `output${idx + 1}`;
          },
        );
      }

      normalizeTabNeuronIo(tab);
      normalizeDatasetRows(tab);
      tab.outputNeuronValues = Array.from({ length: outputCount }, () => "-");
    });
  }

  function openDatasetModal() {
    updateActiveTab((tab) => {
      normalizeDatasetRows(tab);
    });
    datasetModalOpen = true;
    const maxWidth = Math.max(540, window.innerWidth - 20);
    const maxHeight = Math.max(320, window.innerHeight - 20);
    trainingWindowSize = {
      width: Math.min(trainingWindowSize.width, maxWidth),
      height: Math.min(trainingWindowSize.height, maxHeight),
    };
    trainingWindowPosition = {
      x: Math.min(trainingWindowPosition.x, Math.max(10, maxWidth - 100)),
      y: Math.min(trainingWindowPosition.y, Math.max(10, maxHeight - 100)),
    };
  }

  function addDatasetRow() {
    updateActiveTab((tab) => {
      normalizeDatasetRows(tab);
      tab.datasetRows.push({
        input: Array.from({ length: tab.layers[0] }, () => 0),
        target: Array.from(
          { length: tab.layers[tab.layers.length - 1] },
          () => 0,
        ),
      });
    });
  }

  function removeDatasetRow(rowIndex) {
    updateActiveTab((tab) => {
      normalizeDatasetRows(tab);
      if (tab.datasetRows.length <= 1) {
        return;
      }
      tab.datasetRows.splice(rowIndex, 1);
    });
  }

  function setDatasetRowInput(rowIndex, inputIndex, value) {
    updateActiveTab((tab) => {
      normalizeDatasetRows(tab);
      const nextValue = Number(value);
      tab.datasetRows[rowIndex].input[inputIndex] = Number.isFinite(nextValue)
        ? nextValue
        : 0;
    });
  }

  function setDatasetRowOutput(rowIndex, outputIndex, value) {
    updateActiveTab((tab) => {
      normalizeDatasetRows(tab);
      const nextValue = Number(value);
      tab.datasetRows[rowIndex].target[outputIndex] = Number.isFinite(nextValue)
        ? nextValue
        : 0;
    });
  }

  function exportDatasetCsv() {
    const active = getActiveTab();
    const rows = currentDatasetRows(active);
    const inputCount = active.layers[0] ?? 0;
    const outputCount = active.layers[active.layers.length - 1] ?? 0;

    // speichere die anzahl von input/output-neuronen in der ersten zeile
    const nodeCount = "in:" + inputCount + ",out:" + outputCount;

    const lines = [nodeCount];

    // neuron-namen:
    const header = [...active.inputNeuronNames, ...active.outputNeuronNames];
    lines.push(header.join(","));

    for (const row of rows) {
      const values = [
        ...row.input.map((value) => String(value)),
        ...row.target.map((value) => String(value)),
      ];
      lines.push(values.join(","));
    }

    const blob = new Blob([lines.join("\n")], {
      type: "text/csv;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${active.name.replace(/\s+/g, "_")}_trainingsdaten.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function exportCurrentNetwork() {
    const active = getActiveTab();
    const state = active.state
      ? clone(active.state)
      : buildPlaceholderState(active);
    const inputCount = active.layers[0] ?? 0;
    const outputCount = active.layers[active.layers.length - 1] ?? 0;

    const payload = {
      version: 1,
      exported_at: new Date().toISOString(),
      name: active.name,
      training_epochs: activeTab.epochs,
      network: {
        layers: clone(active.layers),
        activation: active.activation,
        learning_rate: Number(active.learningRate),
        show_output_segment: Boolean(active.showOutputSegment),
        input_names: Array.from({ length: inputCount }, (_, idx) =>
          String(active.inputNeuronNames?.[idx] ?? `input${idx + 1}`),
        ),
        output_names: Array.from({ length: outputCount }, (_, idx) =>
          String(active.outputNeuronNames?.[idx] ?? `output${idx + 1}`),
        ),
        input_values: Array.from({ length: inputCount }, (_, idx) =>
          String(active.inputNeuronValues?.[idx] ?? "0"),
        ),
        output_values: Array.from({ length: outputCount }, (_, idx) =>
          String(active.outputNeuronValues?.[idx] ?? "-"),
        ),
        state,
      },
    };

    const text = JSON.stringify(payload, null, 2);
    const blob = new Blob([text], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${active.name.replace(/\s+/g, "_")}_netz.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function triggerImportNetwork() {
    networkImportInputEl?.click();
  }

  function assertLoadedNetworkShape(network) {
    const layers = Array.isArray(network?.layers) ? network.layers : null;
    if (!layers || layers.length < 2) {
      throw new Error("Import abgebrochen: layers fehlt oder ist ungültig.");
    }

    if (!layers.every((n) => Number.isInteger(n) && n > 0)) {
      throw new Error(
        "Import abgebrochen: layers muss nur positive Ganzzahlen enthalten.",
      );
    }

    if (!network?.state || typeof network.state !== "object") {
      throw new Error("Import abgebrochen: state fehlt.");
    }

    if (
      !Array.isArray(network.state.weights) ||
      !Array.isArray(network.state.biases)
    ) {
      throw new Error("Import abgebrochen: state.weights/state.biases fehlt.");
    }
  }

  async function onNetworkFileSelected(event) {
    try {
      const file = event.currentTarget.files?.[0];
      event.currentTarget.value = "";
      if (!file) {
        return;
      }

      const text = await file.text();
      const parsed = JSON.parse(text);
      const network = parsed?.network;

      assertLoadedNetworkShape(network);

      const layers = clone(network.layers);
      const inputCount = layers[0];
      const outputCount = layers[layers.length - 1];

      const inputNames = Array.from({ length: inputCount }, (_, idx) => {
        const raw = String(network.input_names?.[idx] ?? "").trim();
        return raw.length > 0 ? raw : `input${idx + 1}`;
      });

      const outputNames = Array.from({ length: outputCount }, (_, idx) => {
        const raw = String(network.output_names?.[idx] ?? "").trim();
        return raw.length > 0 ? raw : `output${idx + 1}`;
      });

      const inputValues = Array.from({ length: inputCount }, (_, idx) =>
        String(network.input_values?.[idx] ?? "0"),
      );

      const outputValues = Array.from({ length: outputCount }, (_, idx) =>
        String(network.output_values?.[idx] ?? "-"),
      );

      updateActiveTab((tab) => {
        tab.name = String(parsed?.name ?? tab.name);
        const importedEpochs = Number(parsed?.training_epochs);
        tab.epochs =
          Number.isFinite(importedEpochs) && importedEpochs >= 0
            ? Math.floor(importedEpochs)
            : Number(tab.epochs) || 0;
        tab.showOutputSegment = Boolean(
          network.show_output_segment ??
            parsed?.show_output_segment ??
            tab.showOutputSegment,
        );
        tab.layers = layers;
        tab.activation = String(
          network.activation ?? tab.activation ?? "logistic",
        );
        tab.learningRate = Number(
          network.learning_rate ?? tab.learningRate ?? 0.1,
        );
        tab.state = clone(network.state);
        tab.inputNeuronNames = inputNames;
        tab.outputNeuronNames = outputNames;
        tab.inputNeuronValues = inputValues;
        tab.outputNeuronValues = outputValues;
        tab.lossHistory = Array.isArray(network.loss_history)
          ? [...network.loss_history]
          : [];
        normalizeTabNeuronIo(tab);
        normalizeDatasetRows(tab);
      });

      activationMenuOpen = false;
      await runLiveInferenceForTab(activeTabId, { ensureState: false });
    } catch (error) {
      console.log(error instanceof Error ? error.message : String(error));
    }
  }

  async function onDatasetFileSelected(event) {
    try {
      const file = event.currentTarget.files?.[0];
      event.currentTarget.value = "";
      if (!file) {
        return;
      }

      const text = await file.text();
      const lines = text
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

      if (lines.length === 0) {
        throw new Error("CSV-Datei ist leer.");
      }

      const { inputCount, outputCount } = parseNodeCountLine(lines[0]);
      const delimiter = detectCsvDelimiter(
        lines.length > 1 ? lines.slice(1) : lines,
      );

      pendingImportCsvText = text;
      pendingImportFirstLine = lines[0];
      pendingImportSecondLine = lines[1] ?? "(keine zweite Zeile vorhanden)";
      pendingImportInputCount = inputCount;
      pendingImportOutputCount = outputCount;
      pendingImportDelimiter = delimiter;
      datasetImportStep = "ask-second-line";
      datasetImportPromptOpen = true;
    } catch (error) {
      closeDatasetImportPrompt();
      console.error(error instanceof Error ? error.message : String(error));
    }
  }

  function answerImportSecondLineIsNames(isNames) {
    try {
      const lines = pendingImportCsvText
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

      if (!isNames) {
        const parsedRows = parseImportRows(
          lines,
          pendingImportDelimiter,
          1,
          pendingImportInputCount,
          pendingImportOutputCount,
        );
        applyImportedDataset(parsedRows, null, false);
        closeDatasetImportPrompt();
        return;
      }

      const importedNames = parseImportNames(
        lines,
        pendingImportDelimiter,
        pendingImportInputCount,
        pendingImportOutputCount,
      );

      const active = getActiveTab();
      const currentInputNames = Array.from(
        { length: pendingImportInputCount },
        (_, idx) => String(active.inputNeuronNames?.[idx] ?? `input${idx + 1}`),
      );
      const currentOutputNames = Array.from(
        { length: pendingImportOutputCount },
        (_, idx) =>
          String(active.outputNeuronNames?.[idx] ?? `output${idx + 1}`),
      );

      const namesDiffer =
        !arraysEqual(currentInputNames, importedNames.inputNames) ||
        !arraysEqual(currentOutputNames, importedNames.outputNames);

      if (!namesDiffer) {
        const parsedRows = parseImportRows(
          lines,
          pendingImportDelimiter,
          2,
          pendingImportInputCount,
          pendingImportOutputCount,
        );
        applyImportedDataset(parsedRows, importedNames, true);
        closeDatasetImportPrompt();
        return;
      }

      datasetImportStep = "ask-adopt-names";
    } catch (error) {
      closeDatasetImportPrompt();
      console.error(error instanceof Error ? error.message : String(error));
    }
  }

  function answerImportAdoptNames(adoptNames) {
    try {
      const lines = pendingImportCsvText
        .split(/\r?\n/)
        .map((line) => line.trim())
        .filter((line) => line.length > 0);

      const importedNames = parseImportNames(
        lines,
        pendingImportDelimiter,
        pendingImportInputCount,
        pendingImportOutputCount,
      );

      const parsedRows = parseImportRows(
        lines,
        pendingImportDelimiter,
        2,
        pendingImportInputCount,
        pendingImportOutputCount,
      );

      applyImportedDataset(parsedRows, importedNames, adoptNames);
      closeDatasetImportPrompt();
    } catch (error) {
      closeDatasetImportPrompt();
      console.error(error instanceof Error ? error.message : String(error));
    }
  }

  function delay(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  function buildActivationList(layers, activation) {
    return Array.from({ length: layers.length - 1 }, () => activation);
  }

  function buildPlaceholderState(tab) {
    const layers = tab.layers;
    const activations = buildActivationList(layers, tab.activation);
    const weights = [];
    const biases = [];

    for (let layer = 0; layer < layers.length - 1; layer += 1) {
      const inSize = layers[layer];
      const outSize = layers[layer + 1];
      const layerWeights = [];
      for (let to = 0; to < outSize; to += 1) {
        layerWeights.push(Array.from({ length: inSize }, () => 0));
      }
      weights.push(layerWeights);
      biases.push(Array.from({ length: outSize }, () => 0));
    }

    return {
      layers: clone(layers),
      activations,
      learning_rate: Number(tab.learningRate),
      weights,
      biases,
    };
  }

  function geometryFromState(state) {
    const layers = state.layers;
    const width = Math.max(680, 180 * layers.length);
    const maxNodes = Math.max(...layers);
    const height = Math.min(600, Math.max(340, 90 * maxNodes));

    const leftPad = 90;
    const rightPad = 90;
    const topPad = 55;
    const bottomPad = 55;

    const xStep =
      layers.length > 1
        ? (width - leftPad - rightPad) / (layers.length - 1)
        : width / 2;

    const nodes = [];
    const layersById = [];

    for (let layer = 0; layer < layers.length; layer += 1) {
      const count = layers[layer];
      const yStep = count > 1 ? (height - topPad - bottomPad) / (count - 1) : 0;
      const layerNodeIds = [];

      for (let idx = 0; idx < count; idx += 1) {
        const node = {
          id: `l${layer}-n${idx}`,
          layer,
          node: idx,
          x: leftPad + layer * xStep,
          y: count === 1 ? height / 2 : topPad + idx * yStep,
        };
        nodes.push(node);
        layerNodeIds.push(node.id);
      }
      layersById.push(layerNodeIds);
    }

    const nodeById = new Map(nodes.map((n) => [n.id, n]));
    const connections = [];
    const weightLabelTargetRatio = 0.7;

    for (let layer = 1; layer < layers.length; layer += 1) {
      for (let to = 0; to < layers[layer]; to += 1) {
        for (let from = 0; from < layers[layer - 1]; from += 1) {
          const fromNode = nodeById.get(`l${layer - 1}-n${from}`);
          const toNode = nodeById.get(`l${layer}-n${to}`);
          const weight =
            state.weights?.[layer - 1]?.[to]?.[from] !== undefined
              ? state.weights[layer - 1][to][from]
              : 0;
          connections.push({
            id: `w-${layer - 1}-${to}-${from}`,
            layer: layer - 1,
            to,
            from,
            x1: fromNode.x,
            y1: fromNode.y,
            x2: toNode.x,
            y2: toNode.y,
            labelX:
              fromNode.x + (toNode.x - fromNode.x) * weightLabelTargetRatio,
            labelY:
              fromNode.y + (toNode.y - fromNode.y) * weightLabelTargetRatio,
            weight,
          });
        }
      }
    }

    return { width, height, nodes, connections, layersById };
  }

  async function initWasm() {
    if (wasmReady) {
      return;
    }

    await initWorker();
    wasmReady = true;

    const listed = await callWorker("nnListActivations");
    if (Array.isArray(listed?.activations) && listed.activations.length > 0) {
      availableActivations = listed.activations;
    }
  }

  async function requireApi() {
    await initWorker();
  }

  async function withBusy(task) {
    busy = true;
    try {
      await task();
    } catch (error) {
      console.error(error instanceof Error ? error.message : String(error));
    } finally {
      busy = false;
    }
  }

  async function createStateForTab(tabId) {
    await initWasm();
    await requireApi();

    const tab = tabs.find((item) => item.id === tabId);
    if (!tab) {
      throw new Error("Tab nicht gefunden.");
    }

    const payload = {
      layers: tab.layers,
      activations: buildActivationList(tab.layers, tab.activation),
      learning_rate: Number(tab.learningRate),
    };

    const result = await callWorker("nnCreateState", payload);

    updateTab(tabId, (t) => {
      t.state = result.state;
      normalizeTabNeuronIo(t);
      t.outputNeuronValues = Array.from(
        { length: t.layers[t.layers.length - 1] },
        () => "-",
      );
      t.lossHistory = [];
    });

    await runLiveInferenceForTab(tabId, { ensureState: false });
  }

  async function runLiveInferenceForTab(tabId, options = {}) {
    const { ensureState = false } = options;
    const runId = ++liveInferenceRunId;

    try {
      if (!wasmReady) {
        return;
      }

      await requireApi();

      let tab = tabs.find((entry) => entry.id === tabId);
      if (!tab) {
        return;
      }

      if (!tab.state) {
        if (!ensureState) {
          return;
        }
        await createStateForTab(tabId);
        tab = tabs.find((entry) => entry.id === tabId);
        if (!tab?.state) {
          return;
        }
      }

      const payload = {
        state: tab.state,
        input: parseNeuronInputs(tab),
      };

      const result = await callWorker("nnForward", payload);

      if (runId !== liveInferenceRunId) {
        return;
      }

      updateTab(tabId, (next) => {
        normalizeTabNeuronIo(next);
        next.outputNeuronValues = mapOutputsToStrings(next, result.output);
      });
    } catch (error) {
      if (tabId === activeTabId) {
        console.error(error instanceof Error ? error.message : String(error));
      }
    }
  }

  function randomizeActiveState() {
    return withBusy(async () => {
      const active = getActiveTab();
      await createStateForTab(active.id);
      updateTab(active.id, (next) => {
        next.epochs = 0;
      });
    });
  }

  function ensureStateForActiveTab() {
    const active = getActiveTab();
    if (active.state) {
      return Promise.resolve();
    }
    return createStateForTab(active.id);
  }

  function applyTrainingSnapshot(tabId, snapshot) {
    if (!snapshot) {
      return;
    }

    const combinedHistory = Array.isArray(snapshot.loss_history)
      ? [...trainingLossHistoryBase, ...snapshot.loss_history]
      : trainingLossHistoryBase;

    updateTab(tabId, (next) => {
      next.state = snapshot.state;
      next.lossHistory = combinedHistory;
    });

    activeTab.epochs =
      trainingEpochOffset + Number(snapshot.epochs_done ?? activeTab.epochs);
    trainingLastLoss = snapshot.has_final_loss
      ? Number(snapshot.final_loss)
      : trainingLastLoss;
    trainingDeviation = Number(snapshot.deviation ?? trainingDeviation ?? 0);

    if (tabId === activeTabId) {
      runLiveInferenceForTab(tabId, { ensureState: false });
    }
  }

  async function finalizeTrainingSession(
    tabId,
    trainerId,
    fallbackStatus = null,
  ) {
    if (!trainerId) {
      return;
    }

    let snapshotApplied = false;

    if (fallbackStatus) {
      applyTrainingSnapshot(tabId, fallbackStatus);
      snapshotApplied = true;
    }

    try {
      await callWorker("nnTrainerStop", {
        trainer_id: trainerId,
      });
    } catch {
      // No-op: if trainer already stopped/disposed we still try status/dispose.
    }

    try {
      const finalStatus = await callWorker("nnTrainerStatus", {
        trainer_id: trainerId,
      });

      applyTrainingSnapshot(tabId, finalStatus);
      snapshotApplied = true;
    } catch {
      // Best effort: if status fetch fails, keep fallback snapshot.
    }

    if (!snapshotApplied && tabId) {
      // If we have neither fallback nor status, keep existing UI state as-is.
    }

    try {
      await callWorker("nnTrainerDispose", {
        trainer_id: trainerId,
      });
    } catch {
      // No-op.
    }
  }

  async function trainActive() {
    if (isTraining) {
      return;
    }

    let finalTabId = "";
    let finalTrainerId = "";
    let lastObservedStatus = null;

    console.log(activeTab.learningRate);
    // check if Lernrate valid
    if (
      isNaN(Number(activeTab.learningRate)) ||
      Number(activeTab.learningRate) <= 0
    ) {
      alert("Lernrate muss eine Zahl > 0 sein.");
      return;
    }

    try {
      await initWasm();
      await requireApi();
      await ensureStateForActiveTab();

      const active = getActiveTab();
      normalizeDatasetRows(active);
      const dataset = currentDatasetRows(active);
      if (!Array.isArray(dataset) || dataset.length === 0) {
        throw new Error("Trainingsdaten fehlerhaft.");
      }

      const initResult = await callWorker("nnTrainerInit", {
        state: active.state,
        dataset,
        learning_rate: Number(active.learningRate),
        shuffle: Boolean(active.shuffle),
      });

      const trainerId = String(initResult.trainer_id || "");
      if (!trainerId) {
        throw new Error(
          "Training konnte nicht gestartet werden (trainer_id fehlt).",
        );
      }

      isTraining = true;
      stopTrainingRequested = false;
      trainingTabId = active.id;
      trainingTrainerId = trainerId;
      finalTabId = active.id;
      finalTrainerId = trainerId;
      trainingLossHistoryBase = Array.isArray(active.lossHistory)
        ? [...active.lossHistory]
        : [];
      const persistedEpochs = Number(active.epochs);
      trainingEpochOffset =
        Number.isFinite(persistedEpochs) && persistedEpochs >= 0
          ? Math.floor(persistedEpochs)
          : trainingLossHistoryBase.length;
      activeTab.epochs = trainingEpochOffset;
      trainingLastLoss =
        trainingLossHistoryBase.length > 0
          ? Number(trainingLossHistoryBase[trainingLossHistoryBase.length - 1])
          : null;
      trainingDeviation = null;

      updateTab(active.id, (next) => {
        next.trainerId = trainerId;
      });

      await callWorker("nnTrainerStart", { trainer_id: trainerId });

      while (!stopTrainingRequested) {
        const tab = tabs.find((entry) => entry.id === trainingTabId);
        if (!tab) {
          break;
        }

        const trainStatus = await callWorker("nnTrainerStatus", {
          trainer_id: trainingTrainerId,
        });
        lastObservedStatus = trainStatus;

        const currentLoss = trainStatus.has_final_loss
          ? Number(trainStatus.final_loss)
          : null;
        const currentDeviation = Number(trainStatus.deviation ?? 0);
        const currentEpochsDone =
          trainingEpochOffset + Number(trainStatus.epochs_done ?? 0);
        const combinedHistory = Array.isArray(trainStatus.loss_history)
          ? [...trainingLossHistoryBase, ...trainStatus.loss_history]
          : trainingLossHistoryBase;

        updateTab(trainingTabId, (next) => {
          next.state = trainStatus.state;
          next.lossHistory = combinedHistory;
        });

        if (trainingTabId === activeTabId) {
          runLiveInferenceForTab(trainingTabId, { ensureState: false });
        }

        activeTab.epochs = currentEpochsDone;
        trainingLastLoss = currentLoss;
        trainingDeviation = currentDeviation;

        if (currentLoss !== null) {
          status = `${tab.name}: Training Epoche ${activeTab.epochs}, Loss ${currentLoss.toFixed(6)}, Abweichung ${currentDeviation.toFixed(6)}`;
        }

        if (!trainStatus.running) {
          break;
        }

        await delay(50);
      }
    } catch (error) {
      console.error(error instanceof Error ? error.message : String(error));
    } finally {
      await finalizeTrainingSession(
        finalTabId || trainingTabId,
        finalTrainerId || trainingTrainerId,
        lastObservedStatus,
      );

      if (!stopTrainingRequested && trainingDeviation === 0) {
        const tab = tabs.find(
          (entry) => entry.id === (finalTabId || trainingTabId),
        );
      }

      isTraining = false;
      stopTrainingRequested = false;
      trainingTrainerId = "";
      trainingTabId = "";
      trainingEpochOffset = 0;
      trainingLossHistoryBase = [];
    }
  }

  function handleTrainingButtonClick() {
    if (isTraining) {
      stopTrainingRequested = true;
      return;
    }
    return trainActive();
  }

  function addTab() {
    tabCounter += 1;
    const next = createTab(tabCounter);
    tabs = [...tabs, next];
    activeTabId = next.id;
  }

  function activateTab(tabId) {
    activeTabId = tabId;
    runLiveInferenceForTab(tabId, { ensureState: false });
  }

  function closeTab(tabId) {
    if (isTraining && tabId === trainingTabId) {
      stopTrainingRequested = true;
    }

    if (tabs.length === 1) {
      return;
    }
    const idx = tabs.findIndex((t) => t.id === tabId);
    tabs = tabs.filter((t) => t.id !== tabId);

    if (activeTabId === tabId) {
      const nextIdx = Math.max(0, idx - 1);
      activeTabId = tabs[nextIdx].id;
    }
  }

  function beginRename(tab) {
    renamingTabId = tab.id;
    renameDraft = tab.name;

    // focus input
    setTimeout(() => {
      const input = /** @type {HTMLInputElement | null} */ (
        document.querySelector(`.tab-pill.active .rename-input`)
      );
      console.log(input);
      if (input) {
        input.focus();
        input.select();
      }
    }, 0);
  }

  function finishRename() {
    if (!renamingTabId) {
      return;
    }
    const nextName = renameDraft.trim();
    if (nextName) {
      updateTab(renamingTabId, (tab) => {
        tab.name = nextName;
      });
    }
    renamingTabId = "";
    renameDraft = "";
  }

  function setActiveActivation(value) {
    updateActiveTab((tab) => {
      tab.activation = value;
      if (tab.state) {
        tab.state.activations = buildActivationList(tab.state.layers, value);
      }
    });
  }

  function toggleActivationMenu() {
    activationMenuOpen = !activationMenuOpen;
  }

  function toggleInfoMenu() {
    infoMenuOpen = !infoMenuOpen;
  }

  function onInfoMenuFocusOut(event) {
    const next = event.relatedTarget;
    const current = event.currentTarget;

    if (
      !(current instanceof HTMLElement) ||
      !(next instanceof HTMLElement) ||
      !current.contains(next)
    ) {
      infoMenuOpen = false;
    }
  }

  function closeActivationMenu() {
    activationMenuOpen = false;
  }

  function selectActivationFromMenu(value) {
    setActiveActivation(value);
    activationMenuOpen = false;
  }

  function onActivationMenuFocusOut(event) {
    const next = event.relatedTarget;
    const current = event.currentTarget;

    if (
      !(current instanceof HTMLElement) ||
      !(next instanceof HTMLElement) ||
      !current.contains(next)
    ) {
      activationMenuOpen = false;
    }
  }

  function normalizeActivationName(value) {
    return String(value || "")
      .trim()
      .toLowerCase()
      .replace(/\s+/g, "")
      .replace(/-/g, "_");
  }

  function evaluateActivation(name, x) {
    const key = normalizeActivationName(name);

    if (key === "binary" || key === "step" || key === "heaviside") {
      return x >= 0 ? 1 : 0;
    }

    if (key === "logistic" || key === "sigmoid") {
      return 1 / (1 + Math.exp(-x));
    }

    if (key === "relu") {
      return Math.max(0, x);
    }

    if (key === "linear" || key === "identity") {
      return x;
    }

    // Fallback: unbekannte Aktivierung als lineare Funktion darstellen.
    return x;
  }

  function buildActivationPreview(name) {
    const width = 180;
    const height = 112;
    const padLeft = 12;
    const padRight = 8;
    const padTop = 10;
    const padBottom = 12;

    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const xMin = -3;
    const xMax = 3;
    const yMin = -1;
    const yMax = 3;
    const xRange = xMax - xMin;
    const yRange = yMax - yMin;

    const toX = (x) => padLeft + ((x - xMin) / xRange) * plotWidth;
    const toY = (y) => padTop + (1 - (y - yMin) / yRange) * plotHeight;

    const samples = 80;
    const points = [];
    for (let i = 0; i <= samples; i += 1) {
      const x = xMin + (i / samples) * xRange;
      const y = evaluateActivation(name, x);
      const clampedY = Math.max(yMin, Math.min(yMax, y));
      points.push(`${toX(x)},${toY(clampedY)}`);
    }

    return {
      width,
      height,
      xAxisY: toY(0),
      yAxisX: toX(0),
      linePoints: points.join(" "),
    };
  }

  let activationPreviews = $derived(
    availableActivations.map((name) => ({
      name,
      preview: buildActivationPreview(name),
    })),
  );

  let learningRateInvalid = $state(false);

  function setActiveLearningRate(value) {
    updateActiveTab((tab) => {
      if (value.includes(",")) {
        tab.learningRate = value.replace(",", ".");
        value = tab.learningRate;
      }
      if (isNaN(Number(value)) || Number(value) <= 0) {
        learningRateInvalid = true;
        return;
      }
      learningRateInvalid = false;
      if (tab.state) {
        tab.state.learning_rate = Number(value);
      }
    });
  }

  function setLayerCount(layerIndex, count) {
    const nextCount = Math.max(1, Number(count));
    updateActiveTab((tab) => {
      tab.layers[layerIndex] = nextCount;
      tab.state = null;
      tab.lossHistory = [];
      tab.epochs = 0;
      normalizeTabNeuronIo(tab);
      normalizeDatasetRows(tab);
      tab.outputNeuronValues = Array.from(
        { length: tab.layers[tab.layers.length - 1] },
        () => "-",
      );
    });
  }

  function addHiddenLayer() {
    updateActiveTab((tab) => {
      tab.layers.splice(tab.layers.length - 1, 0, 3);
      tab.state = null;
      tab.lossHistory = [];
      tab.epochs = 0;
      normalizeTabNeuronIo(tab);
      normalizeDatasetRows(tab);
      tab.outputNeuronValues = Array.from(
        { length: tab.layers[tab.layers.length - 1] },
        () => "-",
      );
    });
  }

  function removeHiddenLayer() {
    updateActiveTab((tab) => {
      if (tab.layers.length <= 2) {
        return;
      }
      tab.layers.splice(tab.layers.length - 2, 1);
      tab.state = null;
      tab.lossHistory = [];
      tab.epochs = 0;
      normalizeTabNeuronIo(tab);
      normalizeDatasetRows(tab);
      tab.outputNeuronValues = Array.from(
        { length: tab.layers[tab.layers.length - 1] },
        () => "-",
      );
    });
  }

  function setInputNeuronValue(nodeIndex, value) {
    const tabId = activeTabId;

    updateActiveTab((tab) => {
      normalizeTabNeuronIo(tab);
      tab.inputNeuronValues[nodeIndex] = value;
    });

    runLiveInferenceForTab(tabId, { ensureState: true });
  }

  function editInputNeuronName(nodeIndex) {
    const tab = getActiveTab();
    const current = String(
      tab.inputNeuronNames?.[nodeIndex] ?? `input${nodeIndex + 1}`,
    );

    const value = window.prompt(`Name für Input ${nodeIndex + 1}`, current);

    if (value === null) {
      return;
    }

    const nextName = value.trim() || `input${nodeIndex + 1}`;

    updateActiveTab((next) => {
      normalizeTabNeuronIo(next);
      next.inputNeuronNames[nodeIndex] = nextName;
    });
  }

  function editOutputNeuronName(nodeIndex) {
    const tab = getActiveTab();
    const current = String(
      tab.outputNeuronNames?.[nodeIndex] ?? `output${nodeIndex + 1}`,
    );

    if (
      activeTab.showOutputSegment &&
      ["a", "b", "c", "d", "e", "f", "g"].includes(
        tab.outputNeuronNames?.[nodeIndex],
      )
    ) {
      window.alert(
        "In diesem Modus können die Namen der Output-Neuronen nicht bearbeitet werden, da sie zur Darstellung von Segmentanzeigen verwendet werden.",
      );
      return;
    }

    const value = window.prompt(`Name für Output ${nodeIndex + 1}`, current);

    if (value === null) {
      return;
    }

    const nextName = value.trim() || `output${nodeIndex + 1}`;

    updateActiveTab((next) => {
      normalizeTabNeuronIo(next);
      next.outputNeuronNames[nodeIndex] = nextName;
    });
  }

  function setEpochs(value) {
    updateActiveTab((tab) => {
      activeTab.epochs = Math.max(1, Number(value));
    });
  }

  function setShuffle(value) {
    updateActiveTab((tab) => {
      tab.shuffle = value;
    });
  }

  function editWeight(conn) {
    return withBusy(async () => {
      highlightedConnectionId = conn.id;

      if (isTraining) {
        throw new Error(
          "Gewichte können während des Trainings nicht manuell geändert werden.",
        );
      }

      await ensureStateForActiveTab();

      const tab = getActiveTab();
      const current = tab.state.weights[conn.layer][conn.to][conn.from];
      const value = window.prompt(
        `Gewicht Layer ${conn.layer + 1}, To ${conn.to + 1}, From ${conn.from + 1}`,
        String(current),
      );
      if (value === null) {
        return;
      }
      const num = Number(value);
      if (!Number.isFinite(num)) {
        throw new Error("Ungültiger Zahlenwert für Gewicht.");
      }

      updateActiveTab((next) => {
        next.state.weights[conn.layer][conn.to][conn.from] = num;
      });
      status = `${tab.name}: Gewicht angepasst.`;
    });
  }

  function editBias(layerIndex, nodeIndex) {
    if (layerIndex === 0) {
      return;
    }

    return withBusy(async () => {
      if (isTraining) {
        throw new Error(
          "Bias-Werte können während des Trainings nicht manuell geändert werden.",
        );
      }

      await ensureStateForActiveTab();

      const tab = getActiveTab();
      const bLayer = layerIndex - 1;
      const current = tab.state.biases[bLayer][nodeIndex];
      const value = window.prompt(
        `Bias Layer ${layerIndex + 1}, Node ${nodeIndex + 1}`,
        String(current),
      );
      if (value === null) {
        return;
      }
      const num = Number(value);
      if (!Number.isFinite(num)) {
        throw new Error("Ungültiger Zahlenwert für Bias.");
      }

      updateActiveTab((next) => {
        next.state.biases[bLayer][nodeIndex] = num;
      });
      status = `${tab.name}: Bias angepasst.`;
    });
  }

  let activeTab = $derived(
    tabs.find((tab) => tab.id === activeTabId) || tabs[0],
  );

  let stateForDraw = $derived(
    activeTab?.state || buildPlaceholderState(activeTab),
  );

  let graph = $derived(geometryFromState(stateForDraw));

  let orderedConnections = $derived.by(() => {
    if (!graph?.connections?.length) {
      return [];
    }

    const selected = graph.connections.find(
      (conn) => conn.id === highlightedConnectionId,
    );

    if (!selected) {
      return graph.connections;
    }

    return [
      ...graph.connections.filter(
        (conn) => conn.id !== highlightedConnectionId,
      ),
      selected,
    ];
  });

  let hasLoss = $derived(activeTab?.lossHistory?.length > 0);

  function startTrainingWindowDrag(event) {
    if (event.target?.closest("button")) {
      return;
    }

    if (event.button !== 0) {
      return;
    }

    event.preventDefault();
    trainingWindowDragging = true;
    trainingWindowDragOffset = {
      x: event.clientX - trainingWindowPosition.x,
      y: event.clientY - trainingWindowPosition.y,
    };
  }

  function startTrainingWindowResize(event, direction) {
    if (event.button !== 0) {
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    trainingWindowResizing = true;
    trainingWindowResizeDir = direction;
    trainingWindowResizeStart = {
      x: event.clientX,
      y: event.clientY,
      width: trainingWindowSize.width,
      height: trainingWindowSize.height,
      left: trainingWindowPosition.x,
      top: trainingWindowPosition.y,
    };
  }

  function onGlobalMouseMove(event) {
    if (trainingWindowResizing) {
      const minWidth = 540;
      const minHeight = 320;
      const maxWidth = Math.max(minWidth, window.innerWidth - 20);
      const maxHeight = Math.max(minHeight, window.innerHeight - 20);
      const dx = event.clientX - trainingWindowResizeStart.x;
      const dy = event.clientY - trainingWindowResizeStart.y;

      let nextLeft = trainingWindowResizeStart.left;
      let nextTop = trainingWindowResizeStart.top;
      let nextWidth = trainingWindowResizeStart.width;
      let nextHeight = trainingWindowResizeStart.height;

      if (trainingWindowResizeDir.includes("e")) {
        nextWidth = Math.min(
          Math.max(minWidth, trainingWindowResizeStart.width + dx),
          window.innerWidth - trainingWindowResizeStart.left - 10,
        );
      }
      if (trainingWindowResizeDir.includes("s")) {
        nextHeight = Math.min(
          Math.max(minHeight, trainingWindowResizeStart.height + dy),
          window.innerHeight - trainingWindowResizeStart.top - 10,
        );
      }

      trainingWindowSize = {
        width: Math.min(Math.max(minWidth, nextWidth), maxWidth),
        height: Math.min(Math.max(minHeight, nextHeight), maxHeight),
      };

      trainingWindowPosition = {
        x: Math.max(10, Math.min(nextLeft, window.innerWidth - 10 - minWidth)),
        y: Math.max(10, Math.min(nextTop, window.innerHeight - 10 - minHeight)),
      };
      return;
    }

    if (!trainingWindowDragging) {
      return;
    }

    const nextX = event.clientX - trainingWindowDragOffset.x;
    const nextY = event.clientY - trainingWindowDragOffset.y;
    const maxX = Math.max(
      10,
      window.innerWidth - trainingWindowSize.width - 10,
    );
    const maxY = Math.max(
      10,
      window.innerHeight - trainingWindowSize.height - 10,
    );

    trainingWindowPosition = {
      x: Math.min(Math.max(10, nextX), maxX),
      y: Math.min(Math.max(10, nextY), maxY),
    };
  }

  function onGlobalMouseUp() {
    trainingWindowDragging = false;
    trainingWindowResizing = false;
    trainingWindowResizeDir = "";
  }

  let lossChart = $derived.by(() => {
    const history = activeTab?.lossHistory || [];

    const width = 640;
    const height = 220;
    const padLeft = 44;
    const padRight = 12;
    const padTop = 10;
    const padBottom = 28;

    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - padTop - padBottom;

    const yMin = 0;
    const historyMax = history.length > 0 ? Math.max(...history) : 0;
    const targetHeightRatio = 0.85;
    const yMax = historyMax > 0 ? historyMax / targetHeightRatio : 1;
    const yRange = yMax - yMin || 1;

    const xMinEpoch = 1;
    const xMaxEpoch = Math.max(2, history.length);
    const xRange = xMaxEpoch - xMinEpoch || 1;

    const toX = (epoch) => padLeft + ((epoch - xMinEpoch) / xRange) * plotWidth;
    const toY = (value) => padTop + (1 - (value - yMin) / yRange) * plotHeight;

    const linePoints =
      history.length >= 2
        ? history.map((v, i) => `${toX(i + 1)},${toY(v)}`).join(" ")
        : "";

    const yTickCount = 4;
    const yTicks = Array.from({ length: yTickCount + 1 }, (_, i) => {
      const value = yMin + (i / yTickCount) * (yMax - yMin);
      const decimals = yMax < 1 ? 3 : yMax < 10 ? 2 : 1;
      return {
        y: toY(value),
        value,
        label: value.toFixed(decimals),
      };
    });

    // Top tick is intentionally hidden to keep some headroom and a cleaner axis top.
    const visibleYTicks = yTicks.slice(0, -1);
    // Keep axis top aligned with the highest plotted value, so the line never rises above the axis.
    // If no values exist yet, use the same headroom profile as regular data so the axis stays visible.
    const defaultAxisTopValue = yMin + (yMax - yMin) * targetHeightRatio;
    const yAxisTop = toY(historyMax > 0 ? historyMax : defaultAxisTopValue);

    const desiredXTicks = 6;
    const xTicks = [];
    const step = Math.max(
      1,
      Math.ceil((xMaxEpoch - xMinEpoch) / (desiredXTicks - 1)),
    );

    for (let epoch = xMinEpoch; epoch <= xMaxEpoch; epoch += step) {
      xTicks.push({ epoch, x: toX(epoch) });
    }

    if (xTicks[xTicks.length - 1]?.epoch !== xMaxEpoch) {
      xTicks.push({ epoch: xMaxEpoch, x: toX(xMaxEpoch) });
    }

    return {
      width,
      height,
      padLeft,
      padRight,
      padTop,
      padBottom,
      plotWidth,
      plotHeight,
      yTicks: visibleYTicks,
      yAxisTop,
      xTicks,
      linePoints,
      xAxisY: padTop + plotHeight,
      yAxisX: padLeft,
      hasLine: history.length >= 2,
    };
  });

  onMount(() => {
    window.addEventListener("mousemove", onGlobalMouseMove);
    window.addEventListener("mouseup", onGlobalMouseUp);

    withBusy(async () => {
      await initWasm();
      await requireApi();
      await createStateForTab(activeTabId);
      status = "Bereit. Du kannst jetzt pro Tab ein separates Netz bearbeiten.";

      return () => {
        window.removeEventListener("mousemove", onGlobalMouseMove);
        window.removeEventListener("mouseup", onGlobalMouseUp);
        disposeWorker();
      };
    });
  });

  function outputValueToBool(rawValue) {
    const parsed = Number(rawValue);
    return Number.isFinite(parsed) && parsed > 0.5;
  }

  let seg = $derived.by(() => {
    const result = {
      a: false,
      b: false,
      c: false,
      d: false,
      e: false,
      f: false,
      g: false,
    };

    const names = Array.isArray(activeTab?.outputNeuronNames)
      ? activeTab.outputNeuronNames
      : [];
    const values = Array.isArray(activeTab?.outputNeuronValues)
      ? activeTab.outputNeuronValues
      : [];

    for (let idx = 0; idx < names.length; idx += 1) {
      const key = String(names[idx] ?? "")
        .trim()
        .toLowerCase();

      if (!(key in result)) {
        continue;
      }

      result[key] = outputValueToBool(values[idx]);
    }

    return result;
  });

  function handleShowOutputSegmentChange(nextChecked) {
    if (!nextChecked) {
      return;
    }

    // 7 Output-Nodes bereitstellen mit Namen "a" bis "g"
    updateActiveTab((tab) => {
      const outputCount = 7;
      tab.layers[tab.layers.length - 1] = outputCount;
      tab.outputNeuronNames = ["a", "b", "c", "d", "e", "f", "g"];
      tab.state = null;
      tab.lossHistory = [];
      tab.epochs = 0;
      normalizeTabNeuronIo(tab);
      tab.outputNeuronValues = Array.from({ length: outputCount }, () => "-");
    });
  }
</script>

<main class="app-shell">
  <div class="page-info" onfocusout={onInfoMenuFocusOut}>
    <button
      type="button"
      class="page-info-trigger btn-hover"
      aria-haspopup="dialog"
      aria-expanded={infoMenuOpen}
      onclick={toggleInfoMenu}
    >
      info
    </button>

    {#if infoMenuOpen}
      <div
        class="page-info-dropdown"
        role="dialog"
        aria-label="Informationen zur Webseite"
      >
        <p>
          Diese Webseite visualisiert ein Künstliches Neuronales Netzwerk (KNN)
          das du selber konfigurieren und trainieren kannst. Alle wesentlichen
          Daten kannst du selbst anpassen.
        </p>
        <p>
          Die Berechnungen laufen vollständig im Browser! Im Hintergrund wird
          ein in Go geschriebenes WebAssembly-Modul (WASM) verwendet, das
          wesentlich effizienter als JavaScript läuft.
        </p>

        <hr />

        <div class="about">
          <div>
            Quellcode auf <a
              href="https://github.com/tools-info-bw-de/nn"
              target="_blank"
              rel="noopener">GitHub</a
            >
          </div>

          <div>
            Lizenz:
            <a
              href="https://github.com/tools-info-bw-de/nn/blob/main/LICENSE"
              target="_blank"
              rel="noopener">GPL-3.0</a
            >
          </div>

          <div>Marco Kümmel</div>
        </div>
      </div>
    {/if}
  </div>

  <header class="tabs-header">
    <div class="tab-row">
      {#each tabs as tab}
        <div class={`tab-pill ${tab.id === activeTabId ? "active" : ""}`}>
          <button class="tab-open" onclick={() => activateTab(tab.id)}>
            {#if renamingTabId === tab.id}
              <input
                class="rename-input"
                bind:value={renameDraft}
                onblur={finishRename}
                onkeydown={(e) => {
                  if (e.key === "Enter") {
                    finishRename();
                  }
                  if (e.key === "Escape") {
                    renamingTabId = "";
                  }
                }}
              />
            {:else}
              <span role="button" tabindex="0">
                {tab.name}
              </span>
            {/if}
          </button>
          {#if tab.id === activeTabId}
            <button class="tab-edit btn-hover" onclick={() => beginRename(tab)}
              >✎</button
            >
          {/if}
          <button
            class="tab-close btn-hover"
            onclick={() => closeTab(tab.id)}
            disabled={tabs.length === 1}>×</button
          >
        </div>
      {/each}
      <button class="tab-add btn-hover" onclick={addTab}>+ Netz</button>
    </div>
  </header>

  <section class="toolbar">
    <div class="toolbar-group">
      <label>
        <div class="tooltip">
          Aktivierungsfunktion
          <img
            src={publicAsset("circle-question-solid-full.svg")}
            alt="Question"
            width="18"
            height="18"
          />
          <span class="tooltiptext"
            >Die Aktivierungsfunktion bestimmt, wie die Summe der gewichteten
            Inputs (x-Achse) in die Ausgabe eines Neurons umgewandelt wird
            (y-Achse).<br />
          </span>
        </div>
        <div
          class="activation-select-wrap"
          onfocusout={onActivationMenuFocusOut}
        >
          <button
            type="button"
            class="activation-select-trigger"
            aria-haspopup="listbox"
            aria-expanded={activationMenuOpen}
            onclick={toggleActivationMenu}
          >
            <span>{activeTab.activation}</span>
            <span aria-hidden="true">▾</span>
          </button>

          {#if activationMenuOpen}
            <div
              class="activation-preview-popover"
              role="listbox"
              aria-label="Aktivierungsfunktion auswählen"
            >
              {#each activationPreviews as item}
                <button
                  type="button"
                  role="option"
                  aria-selected={activeTab.activation === item.name}
                  class={`activation-preview-item ${activeTab.activation === item.name ? "active" : ""}`}
                  onclick={() => selectActivationFromMenu(item.name)}
                >
                  <span class="activation-name">{item.name}</span>
                  <svg
                    class="activation-preview"
                    viewBox={`0 0 ${item.preview.width} ${item.preview.height}`}
                    aria-hidden="true"
                  >
                    <line
                      class="axis"
                      x1="0"
                      y1={item.preview.xAxisY}
                      x2={item.preview.width}
                      y2={item.preview.xAxisY}
                    ></line>
                    <line
                      class="axis"
                      x1={item.preview.yAxisX}
                      y1="0"
                      x2={item.preview.yAxisX}
                      y2={item.preview.height}
                    ></line>
                    <polyline
                      class="curve"
                      points={item.preview.linePoints}
                      fill="none"
                    ></polyline>
                  </svg>
                </button>
              {/each}
            </div>
          {/if}
        </div>
      </label>

      <label>
        <div class="tooltip">
          Lernrate
          <img
            src={publicAsset("circle-question-solid-full.svg")}
            alt=""
            width="18"
            height="18"
          />
          <span class="tooltiptext"
            >Die Lernrate bestimmt, wie stark die Gewichte im Netzwerk bei jedem
            Trainingsdurchlauf angepasst werden.<br />Achtung: Lernraten können
            zu hoch oder auch zu niedrig sein, um die Fehlerwerte zu minimieren.</span
          >
        </div>
        <input
          bind:value={activeTab.learningRate}
          class={learningRateInvalid ? "invalid" : ""}
          oninput={(e) => setActiveLearningRate(e.currentTarget.value)}
        />
      </label><!--           inputmode="numeric"
 -->
    </div>

    <div class="toolbar-group">
      <button
        class="btn-hover"
        onclick={openDatasetModal}
        disabled={isTraining}
      >
        <img
          src={publicAsset("person-chalkboard-solid-full.svg")}
          alt=""
          width="16"
          height="16"
        />
        <span>Trainingsdaten</span>
      </button>
      <hr />
      <button
        class="btn-hover"
        onclick={randomizeActiveState}
        disabled={busy || isTraining}
      >
        <img
          src={publicAsset("dice-solid-full.svg")}
          alt=""
          width="16"
          height="16"
        />
        <span>Netz randomisieren</span>
      </button>

      <button
        class="btn-hover {isTraining ? 'btn-is-training' : ''}"
        onclick={handleTrainingButtonClick}
        disabled={busy}
      >
        {#if isTraining}
          <img
            src={publicAsset("stop-solid-full.svg")}
            alt=""
            width="16"
            height="16"
          />
          <span>Abbrechen</span>
        {:else}
          <img
            src={publicAsset("play-solid-full.svg")}
            alt=""
            width="16"
            height="16"
          />
          <span
            >Training
            {activeTab.epochs > 0 ? "fortsetzen" : "starten"}
          </span>
        {/if}
      </button>
    </div>

    <div class="loss-meta">
      <div class="header">
        <div class={isTraining ? "training" : ""}>
          {isTraining ? "Training läuft." : "Training beendet."}
        </div>
        {#if isTraining}
          <span class="loader"></span>
        {/if}
      </div>

      <div class="epochs tooltip">
        Epochen <img
          src={publicAsset("circle-question-solid-full.svg")}
          alt=""
          width="18"
          height="18"
        />: {activeTab.epochs}
        <span class="tooltiptext"
          >Jeder vollständige Trainingsdurchlauf durch alle Trainingsdaten ist
          eine Epoche.</span
        >
      </div>

      <div class="values">
        <div class="tooltip">
          <h4>
            Fehlerwerte <img
              src={publicAsset("circle-question-solid-full.svg")}
              alt="Question"
              width="18"
              height="18"
            />
          </h4>
          <span class="tooltiptext"
            >Die Fehlerwerte zeigen die Abweichung zwischen den erwarteten
            Ergebnissen (Trainingsdaten) und den tatsächlichen Werten des Netzes
            an.</span
          >
        </div>
        <div>
          Max:
          {#if hasLoss}
            {Math.max(...activeTab.lossHistory).toFixed(6)}
          {:else}
            ---
          {/if}
        </div>
        <div>
          Min:
          {#if hasLoss}
            {Math.min(...activeTab.lossHistory).toFixed(6)}
          {:else}
            ---
          {/if}
        </div>
        <div class="last-loss">
          Letzter:
          {#if hasLoss}
            {activeTab.lossHistory[activeTab.lossHistory.length - 1].toFixed(6)}
          {:else}
            ---
          {/if}
        </div>
      </div>
    </div>

    <div>
      <FehlerwertChart {lossChart} />
    </div>
  </section>

  <section class="network-graph-wrap">
    <div class="network-title">
      <h2>Live-Netzansicht</h2>
      <div>(Klicken zum Bearbeiten)</div>
    </div>
    <div class="layer-controls">
      <div class="button-group">
        <span>Hidden Layer:</span>

        <button
          class="btn-hover"
          onclick={removeHiddenLayer}
          disabled={isTraining || activeTab.layers.length <= 2}
        >
          -
        </button>
        <button class="btn-hover" onclick={addHiddenLayer} disabled={isTraining}
          >+
        </button>
      </div>

      <input
        bind:this={networkImportInputEl}
        type="file"
        accept="application/json,.json"
        style="display:none"
        onchange={onNetworkFileSelected}
      />
      {#each activeTab.layers as count, layerIndex}
        <label>
          {layerIndex === 0
            ? "Input"
            : layerIndex === activeTab.layers.length - 1
              ? "Output"
              : `Hidden ${layerIndex}`}
          <input
            type="number"
            min="1"
            step="1"
            value={count}
            disabled={isTraining}
            oninput={(e) => setLayerCount(layerIndex, e.currentTarget.value)}
          />
        </label>
      {/each}

      <div class="button-group save-network-group">
        <span>Netzwerk:</span>

        <button
          class="btn-hover"
          onclick={exportCurrentNetwork}
          disabled={isTraining}
        >
          <img
            src={publicAsset("floppy-disk-solid-full.svg")}
            alt=""
            width="20"
            height="20"
          />
          Speichern</button
        >
        <button
          class="btn-hover"
          onclick={triggerImportNetwork}
          disabled={isTraining}
        >
          <img
            src={publicAsset("upload-solid-full.svg")}
            alt=""
            width="20"
            height="20"
          />
          Öffnen</button
        >
      </div>
    </div>
    <div class="graph-scroll">
      <div class="output-segment-option">
        <input
          type="checkbox"
          bind:checked={activeTab.showOutputSegment}
          id="showOutputSegment"
          onchange={(e) =>
            handleShowOutputSegmentChange(e.currentTarget.checked)}
        />
        <label for="showOutputSegment"> 7-Segment-Anzeige </label>
      </div>

      <div class="graph-area">
        <NetworkGraph
          {graph}
          {orderedConnections}
          {highlightedConnectionId}
          {activeTab}
          {setInputNeuronValue}
          {editInputNeuronName}
          {editOutputNeuronName}
          {editWeight}
          {editBias}
        />

        {#if activeTab.showOutputSegment}
          <div class="output-segment">
            <SevenSegment
              a={seg.a}
              b={seg.b}
              c={seg.c}
              d={seg.d}
              e={seg.e}
              f={seg.f}
              g={seg.g}
              editable={false}
            />
          </div>
        {/if}
      </div>
    </div>
  </section>

  {#if datasetModalOpen}
    <div class="modal-backdrop">
      <TrainingsModal
        {activeTab}
        {trainingWindowPosition}
        {trainingWindowSize}
        {datasetImportPromptOpen}
        {datasetImportStep}
        {pendingImportFirstLine}
        {pendingImportSecondLine}
        {setDatasetRowInput}
        {setDatasetRowOutput}
        {editInputNeuronName}
        {editOutputNeuronName}
        {addDatasetRow}
        {removeDatasetRow}
        {exportDatasetCsv}
        {onDatasetFileSelected}
        {answerImportSecondLineIsNames}
        {answerImportAdoptNames}
        {closeDatasetImportPrompt}
        {startTrainingWindowDrag}
        {startTrainingWindowResize}
        bind:trainingImportError
        bind:datasetModalOpen
        showOutputSegment={activeTab.showOutputSegment}
      />
    </div>
  {/if}
</main>

<style>
  @import url("https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap");

  .invalid {
    outline: 2px solid var(--danger);
  }

  .network-title {
    display: inline-flex;
    align-items: baseline;
    gap: 1rem;
    font-family: var(--font-head);
    font-size: 20px;
  }

  .tooltip,
  .tooltip > h4 {
    display: inline-flex;
  }

  .tooltip {
    position: relative;
  }

  .tooltiptext {
    visibility: hidden;
    width: 250px;
    background-color: black;
    color: #ffffff;
    text-align: center;
    border-radius: 6px;
    padding: 5px;
    position: absolute;
    z-index: 1;
    bottom: 150%;
    left: 50%;
    margin-left: -60px;
  }

  .tooltiptext::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    margin-left: -5px;
    border-width: 5px;
    border-style: solid;
    border-color: black transparent transparent transparent;
  }

  .tooltip:hover .tooltiptext {
    visibility: visible;
  }

  .graph-area {
    border: 1px dashed var(--line);
    display: flex;
    align-items: center;
  }

  .output-segment {
    width: 120px;
    height: 180px;
  }

  .output-segment-option {
    position: absolute;
    top: 0;
    right: 0;
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 4px;
    padding: 0.5rem;
  }

  .output-segment-option > input {
    width: inherit;
  }

  .save-network-group {
    margin-left: auto;
  }

  :global(.button-group) {
    display: flex;
    align-items: center;
    border-radius: 10px;
    height: 40px;
  }

  :global(.button-group > span) {
    background-color: lightgray;
    height: 100%;
    align-content: center;
    padding-left: 5px;
    padding-right: 5px;
  }

  :global(.button-group > *:first-child) {
    border-top-left-radius: 10px;
    border-bottom-left-radius: 10px;
  }

  :global(.button-group > *) {
    border-radius: 0;
    display: flex;
    align-items: center;
    height: 100%;
    font-size: 18px;
  }

  :global(.button-group > *:last-child) {
    border-top-right-radius: 10px;
    border-bottom-right-radius: 10px;
  }

  :root {
    --bg: #f2efe7;
    --ink: #1f2937;
    --surface: rgba(255, 255, 255, 0.76);
    --line: #d2c9b9;
    --accent: #006d77;
    --accent-2: #e76f51;
    --ok: #2a9d8f;
    --danger: #b42318;

    --font-head: "Chakra Petch", sans-serif;
    --font-ui: "IBM Plex Mono", monospace;
  }

  * {
    box-sizing: border-box;
  }

  :global(html),
  :global(body) {
    margin: 0;
    min-height: 100%;
    color: var(--ink);
    background: radial-gradient(
        circle at 12% 12%,
        rgba(0, 109, 119, 0.2),
        transparent 26%
      ),
      radial-gradient(
        circle at 88% 16%,
        rgba(231, 111, 81, 0.2),
        transparent 28%
      ),
      linear-gradient(160deg, #f7f4ec 0%, #ebe5d8 64%, #f5f1e8 100%);
    font-family: var(--font-ui);
  }

  :global(#app) {
    min-height: 100svh;
  }

  .app-shell {
    width: min(1280px, 95vw);
    margin: 0 auto;
    padding: 1.4rem 0 2rem;
    display: grid;
    gap: 1rem;
    animation: rise-in 450ms ease-out;
  }

  .tabs-header,
  .toolbar,
  .network-controls,
  .network-graph-wrap {
    background: var(--surface);
    border: 1px solid var(--line);
    border-radius: 14px;
    backdrop-filter: blur(8px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.07);
    padding: 0.85rem;
  }

  .tabs-header {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
  }

  .tab-row > div,
  .tab-row > button {
    margin-top: 0.85rem;
    margin-bottom: 0.85rem;
  }

  .tab-row {
    display: flex;
    gap: 0.45rem;
    align-items: center;
    overflow-x: auto;
    padding-bottom: 0.2rem;
  }

  .tab-pill {
    display: flex;
    align-items: center;
    gap: 0.25rem;
    border: 1px solid var(--line);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.82);
    min-height: 2.1rem;
  }

  .tab-pill.active {
    border-color: var(--accent);
    box-shadow: inset 0 0 0 1px rgba(0, 109, 119, 0.25);
    background-color: rgba(228, 245, 245, 0.8);
  }

  .tab-open,
  .tab-edit,
  .tab-close,
  .tab-add {
    border: 0;
    background: transparent;
    cursor: pointer;
    font: inherit;
    color: inherit;
  }

  .tab-open {
    padding: 0.35rem 0.75rem;
    white-space: nowrap;
  }

  .tab-edit,
  .tab-close {
    width: 1.8rem;
    height: 1.8rem;
    border-radius: 999px;
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .tab-edit:hover,
  .tab-close:hover {
    background: rgba(0, 0, 0, 0.06);
  }

  .tab-add {
    margin-left: 0.3rem;
    border: 1px dashed var(--line);
    border-radius: 999px;
    padding: 0.4rem 0.9rem;
  }

  .rename-input {
    width: 8rem;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.92);
    padding: 0.1rem 0.45rem;
    font: inherit;
  }

  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 0.7rem;
    align-items: start;
    justify-content: space-around;
  }

  .toolbar-group {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    height: 100%;
    justify-content: space-around;
  }

  .activation-select-wrap {
    position: relative;
  }

  .activation-select-trigger {
    width: 100%;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.9);
    color: var(--ink);
    padding: 0.5rem 0.55rem;
    font-weight: 500;
  }

  .activation-preview-popover {
    position: absolute;
    left: 0;
    top: calc(100% + 0.4rem);
    z-index: 40;
    display: flex;
    gap: 0.55rem;
    align-items: stretch;
    overflow-x: auto;
    max-width: min(72vw, 640px);
    padding: 0.55rem;
    border: 1px solid var(--line);
    border-radius: 12px;
    background: #fbfaf5;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.18);
  }

  .activation-preview-item {
    display: flex;
    flex-direction: column;
    align-items: stretch;
    min-width: 170px;
    padding: 0.35rem;
    border-radius: 12px;
    border: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.88);
    color: var(--ink);
    gap: 0.25rem;
    text-align: left;
  }

  .activation-preview-item:hover {
    border-color: var(--accent);
    box-shadow: inset 0 0 0 1px rgba(0, 109, 119, 0.15);
    background: rgba(255, 255, 255, 0.96);
  }

  .activation-preview-item.active {
    border-color: var(--accent);
    box-shadow: inset 0 0 0 1px rgba(0, 109, 119, 0.25);
    background: rgba(228, 245, 245, 0.8);
  }

  .activation-name {
    text-align: center;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.03em;
    font-size: 0.74rem;
  }

  .activation-preview {
    width: 100%;
    height: 90px;
    border: 1px solid var(--line);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.72);
  }

  .activation-preview .axis {
    stroke: rgba(31, 41, 55, 0.45);
    stroke-width: 1;
  }

  .activation-preview .curve {
    stroke: var(--accent-2);
    stroke-width: 2;
  }

  .page-info {
    position: fixed;
    top: 0.8rem;
    right: 0.8rem;
    z-index: 200;
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.45rem;
  }

  .page-info-trigger {
    min-width: 4.8rem;
    text-align: center;
  }

  .page-info-dropdown {
    width: min(360px, calc(100vw - 1.6rem));
    border: 1px solid var(--line);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.2);
    padding: 0.7rem 0.8rem;
  }

  .page-info-dropdown p {
    margin: 0.35rem 0;
    font-size: 0.88rem;
    line-height: 1.4;
  }

  .about {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.35rem;
  }

  hr {
    width: 80%;
  }

  .toolbar-group > button {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
  }

  .btn-is-training {
    background: linear-gradient(135deg, #b42318, #7f1d1d);
  }

  :global(button > img, label > img) {
    filter: invert(1);
    width: 20px;
    margin-right: auto;
  }

  .toolbar-group > button > span {
    margin-right: auto;
  }

  label {
    display: grid;
    gap: 0.35rem;
    font-size: 0.8rem;
  }

  input,
  button {
    font: inherit;
  }

  :global(input) {
    width: 100%;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.9);
    color: var(--ink);
    padding: 0.5rem 0.55rem;
  }

  .layer-controls input {
    width: 6rem;
  }

  :global(button) {
    border: 0;
    border-radius: 10px;
    padding: 0.62rem 0.8rem;
    background: linear-gradient(135deg, var(--accent), #0b8f9b);
    color: #f8fffe;
    font-weight: 500;
    cursor: pointer;
  }

  :global(.btn-hover) {
    transition: 200ms ease;
  }

  :global(.btn-hover:hover) {
    box-shadow: 0 4px 10px rgba(0, 109, 119, 0.4);
    filter: brightness(1.2);
  }

  :global(button:disabled) {
    cursor: not-allowed;
    opacity: 0.7;
    transform: none;
  }

  .network-graph-wrap h2 {
    margin: 0 0 0.7rem;
    font-family: var(--font-head);
  }

  .layer-controls {
    display: flex;
    gap: 0.7rem;
    flex-wrap: wrap;
    margin-bottom: 0.5rem;
    align-items: center;
  }

  .graph-scroll {
    overflow-x: hidden;
    position: relative;
  }

  .hint {
    margin: 0 0 0.7rem;
    font-size: 0.8rem;
    opacity: 0.8;
  }

  .error {
    margin-top: 0.4rem;
    color: var(--danger);
  }

  :global(.modal-backdrop) {
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 42, 0.25);
    z-index: 100;
    pointer-events: none;
  }

  :global(.modal-window) {
    position: fixed;
    width: min(760px, 100%);
    background: #fbfaf5;
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 0.9rem;
    pointer-events: auto;
  }

  :global(.modal-head) {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
  }

  :global(.modal-title) {
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 600;
  }

  :global(.modal-drag-handle) {
    margin: 0;
    font-family: var(--font-head);
    font-size: 1.1rem;
    font-weight: 600;
    cursor: move;
    user-select: none;
  }

  .loader {
    width: 16px;
    height: 16px;
    border: 3px solid #000000;
    border-bottom-color: transparent;
    border-radius: 50%;
    display: inline-block;
    box-sizing: border-box;
    animation: rotation 1s linear infinite;
  }

  @keyframes rotation {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }

  .loss-meta {
    margin: 0.6rem 0 0;
    font-size: 0.82rem;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }

  .loss-meta > .header {
    display: flex;
    align-items: center;
    gap: 0.45rem;
  }

  .loss-meta > .header > .training {
    font-weight: 600;
    color: var(--accent);
    text-decoration: underline;
    text-decoration-color: var(--accent);
  }

  .loss-meta .last-loss {
    font-weight: 600;
  }

  .values h4 {
    margin: 0;
    text-decoration: underline;
  }

  .resizable-window {
    position: fixed;
  }

  :global(.btn-file) {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border: 0;
    border-radius: 10px;
    padding: 0.62rem 0.8rem;
    background: linear-gradient(135deg, var(--accent), #0b8f9b);
    color: #f8fffe;
    font-weight: 500;
    cursor: pointer;
  }

  :global(.btn-file input) {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
  }

  .import-modal {
    width: min(640px, 95vw);
  }

  .csv-first-line {
    font-family: var(--font-ui);
    font-size: 0.82rem;
    background: rgba(0, 0, 0, 0.04);
    border: 1px solid var(--line);
    border-radius: 8px;
    padding: 0.5rem 0.55rem;
    word-break: break-all;
  }

  .import-actions {
    display: flex;
    gap: 0.55rem;
    flex-wrap: wrap;
  }

  @media (max-width: 760px) {
    .app-shell {
      width: 96vw;
    }

    .layer-controls {
      grid-template-columns: 1fr;
    }

    :global(.button-group) {
      width: 100%;
    }

    :global(.button-group button) {
      flex: 1;
    }
  }

  @keyframes rise-in {
    from {
      opacity: 0;
      transform: translateY(8px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
</style>
