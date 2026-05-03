<script>
  import { onMount } from "svelte";
  import FehlerwertChart from "./lib/FehlerwertChart.svelte";
  import NetworkGraph from "./lib/NetworkGraph.svelte";
  import TrainingsModal from "./lib/TrainingsModal.svelte";

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

  let wasmReady = false;
  let busy = $state(false);
  let status = $state("Initialisiere...");
  let errorText = $state("");

  let availableActivations = $state(["binary", "logistic", "relu"]);

  let tabCounter = 1;
  let tabs = $state([createTab(tabCounter)]);
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
  let pendingImportCsvText = "";
  let pendingImportFirstLine = $state("");

  let isTraining = $state(false);
  let stopTrainingRequested = false;
  let trainingTabId = "";
  let trainingTrainerId = "";
  let trainingEpochOffset = 0;
  let trainingLossHistoryBase = [];
  let trainingEpochsDone = $state(0);
  let trainingLastLoss = null;
  let trainingDeviation = null;
  let highlightedConnectionId = $state("");
  let liveInferenceRunId = 0;
  let nnWorker = null;
  let nnWorkerReadyPromise = null;
  let nnWorkerRequestId = 0;
  const nnWorkerPending = new Map();

  function initWorker() {
    if (nnWorkerReadyPromise) {
      return nnWorkerReadyPromise;
    }

    nnWorkerReadyPromise = new Promise((resolve, reject) => {
      const worker = new Worker(new URL("./lib/nnWorker.js", import.meta.url));
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
        const err = new Error(event.message || "WASM-Worker abgestuerzt.");
        reject(err);
      };

      worker.postMessage({ type: "init", id: initRequestId });
    });

    return nnWorkerReadyPromise;
  }

  async function callWorker(method, payload = {}) {
    await initWorker();

    if (!nnWorker) {
      throw new Error("WASM-Worker nicht verfuegbar.");
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

    if (nnWorker) {
      nnWorker.terminate();
      nnWorker = null;
    }
    nnWorkerReadyPromise = null;
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
      epochs: 200,
      shuffle: true,
      datasetRows: defaultRows,
      inputNeuronValues: Array.from({ length: layers[0] }, (_, idx) =>
        String(defaultInputValues[idx] ?? 0),
      ),
      outputNeuronValues: Array.from(
        { length: layers[layers.length - 1] },
        () => "-",
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

    const prevOutputs = Array.isArray(tab.outputNeuronValues)
      ? tab.outputNeuronValues
      : [];

    tab.outputNeuronValues = Array.from(
      { length: outputCount },
      (_, idx) => prevOutputs[idx] ?? "-",
    );
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

    const header = [
      ...Array.from({ length: outputCount }, (_, idx) => `output_${idx + 1}`),
      ...Array.from({ length: inputCount }, (_, idx) => `input_${idx + 1}`),
    ];

    const lines = [header.join(",")];

    for (const row of rows) {
      const values = [
        ...row.target.map((value) => String(value)),
        ...row.input.map((value) => String(value)),
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

  async function onDatasetFileSelected(event) {
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

    pendingImportCsvText = text;
    pendingImportFirstLine = lines[0];
    datasetImportPromptOpen = true;
  }

  function closeDatasetImportPrompt() {
    datasetImportPromptOpen = false;
    pendingImportCsvText = "";
    pendingImportFirstLine = "";
  }

  function importDatasetCsv(hasHeader) {
    const active = getActiveTab();
    const inputCount = active.layers[0] ?? 0;
    const outputCount = active.layers[active.layers.length - 1] ?? 0;
    const neededColumns = inputCount + outputCount;

    const lines = pendingImportCsvText
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    if (lines.length === 0) {
      throw new Error("CSV-Datei ist leer.");
    }

    const delimiter = detectCsvDelimiter(lines);
    const allRows = lines.map((line) => splitCsvLine(line, delimiter));
    const dataRows = hasHeader ? allRows.slice(1) : allRows;

    if (dataRows.length === 0) {
      throw new Error("Keine Datenzeilen in CSV gefunden.");
    }

    const parsedRows = dataRows.map((row, rowIndex) => {
      if (row.length < neededColumns) {
        throw new Error(
          `CSV-Zeile ${rowIndex + 1} hat zu wenige Spalten (erwartet ${neededColumns}).`,
        );
      }

      const target = Array.from({ length: outputCount }, (_, idx) => {
        const value = Number(row[idx]);
        if (!Number.isFinite(value)) {
          throw new Error(
            `Ungueltiger Output-Wert in CSV-Zeile ${rowIndex + 1}.`,
          );
        }
        return value;
      });

      const input = Array.from({ length: inputCount }, (_, idx) => {
        const value = Number(row[outputCount + idx]);
        if (!Number.isFinite(value)) {
          throw new Error(
            `Ungueltiger Input-Wert in CSV-Zeile ${rowIndex + 1}.`,
          );
        }
        return value;
      });

      return { input, target };
    });

    updateActiveTab((tab) => {
      tab.datasetRows = parsedRows;
    });

    closeDatasetImportPrompt();
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

  function currentState() {
    const tab = getActiveTab();
    return tab.state ? tab.state : buildPlaceholderState(tab);
  }

  async function computeMaxDeviation(state, dataset) {
    let maxDeviation = 0;

    for (const sample of dataset) {
      const response = await callWorker("nnForward", {
        state,
        input: sample.input,
      });

      for (let i = 0; i < sample.target.length; i += 1) {
        const outputValue = Number(response.output?.[i] ?? 0);
        const expectedValue = Number(sample.target[i]);
        const diff = Math.abs(outputValue - expectedValue);
        if (diff > maxDeviation) {
          maxDeviation = diff;
        }
      }
    }

    return maxDeviation;
  }

  function geometryFromState(state) {
    const layers = state.layers;
    const width = Math.max(680, 180 * layers.length);
    const maxNodes = Math.max(...layers);
    const height = Math.max(340, 90 * maxNodes);

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
    errorText = "";
    busy = true;
    try {
      await task();
    } catch (error) {
      errorText = error instanceof Error ? error.message : String(error);
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
        errorText = error instanceof Error ? error.message : String(error);
      }
    }
  }

  function randomizeActiveState() {
    return withBusy(async () => {
      const active = getActiveTab();
      await createStateForTab(active.id);
      status = `${active.name}: Gewichte neu randomisiert.`;
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

    trainingEpochsDone =
      trainingEpochOffset + Number(snapshot.epochs_done ?? trainingEpochsDone);
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

    errorText = "";
    let finalTabId = "";
    let finalTrainerId = "";
    let lastObservedStatus = null;

    try {
      await initWasm();
      await requireApi();
      await ensureStateForActiveTab();

      const active = getActiveTab();
      normalizeDatasetRows(active);
      const dataset = currentDatasetRows(active);
      if (!Array.isArray(dataset) || dataset.length === 0) {
        throw new Error(
          "Trainingsdaten muessen ein nicht-leeres JSON-Array sein.",
        );
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
      trainingEpochOffset = trainingLossHistoryBase.length;
      trainingEpochsDone = trainingEpochOffset;
      trainingLastLoss =
        trainingLossHistoryBase.length > 0
          ? Number(trainingLossHistoryBase[trainingLossHistoryBase.length - 1])
          : null;
      trainingDeviation = null;
      status = `${active.name}: Training gestartet.`;

      updateTab(active.id, (next) => {
        next.trainerId = trainerId;
      });

      await callWorker("nnTrainerStart", { trainer_id: trainerId });

      while (!stopTrainingRequested) {
        const tab = tabs.find((entry) => entry.id === trainingTabId);
        if (!tab) {
          status = "Training gestoppt: Trainings-Tab wurde geschlossen.";
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

        trainingEpochsDone = currentEpochsDone;
        trainingLastLoss = currentLoss;
        trainingDeviation = currentDeviation;

        if (currentLoss !== null) {
          status = `${tab.name}: Training Epoche ${trainingEpochsDone}, Loss ${currentLoss.toFixed(6)}, Abweichung ${currentDeviation.toFixed(6)}`;
        }

        if (!trainStatus.running) {
          break;
        }

        await delay(50);
      }

      if (stopTrainingRequested) {
        const tab = tabs.find((entry) => entry.id === trainingTabId);
        status = `${tab?.name ?? "Netz"}: Training manuell abgebrochen.`;
      } else if (trainingDeviation === 0) {
        const tab = tabs.find((entry) => entry.id === trainingTabId);
        status = `${tab?.name ?? "Netz"}: Ziel erreicht (Abweichung 0).`;
      }
    } catch (error) {
      errorText = error instanceof Error ? error.message : String(error);
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
        status = `${tab?.name ?? "Netz"}: Ziel erreicht (Abweichung 0).`;
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
      status = "Stop angefordert...";
      return;
    }
    return trainActive();
  }

  function addTab() {
    tabCounter += 1;
    const next = createTab(tabCounter);
    tabs = [...tabs, next];
    activeTabId = next.id;
    status = `${next.name} erstellt.`;
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

  function setActiveLearningRate(value) {
    updateActiveTab((tab) => {
      tab.learningRate = Number(value);
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

  function setEpochs(value) {
    updateActiveTab((tab) => {
      tab.epochs = Math.max(1, Number(value));
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
          "Gewichte koennen waehrend des Trainings nicht manuell geaendert werden.",
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
        throw new Error("Ungueltiger Zahlenwert fuer Gewicht.");
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
          "Bias-Werte koennen waehrend des Trainings nicht manuell geaendert werden.",
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
        throw new Error("Ungueltiger Zahlenwert fuer Bias.");
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
</script>

<main class="app-shell">
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
          <button class="tab-edit btn-hover" onclick={() => beginRename(tab)}
            >✎</button
          >
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
        Aktivierung
        <select
          value={activeTab.activation}
          onchange={(e) => setActiveActivation(e.currentTarget.value)}
        >
          {#each availableActivations as act}
            <option value={act}>{act}</option>
          {/each}
        </select>
      </label>

      <label>
        Lernrate
        <input
          type="number"
          min="0.001"
          step="0.001"
          value={activeTab.learningRate}
          oninput={(e) => setActiveLearningRate(e.currentTarget.value)}
        />
      </label>
    </div>

    <div class="toolbar-group">
      <button
        class="btn-hover"
        onclick={openDatasetModal}
        disabled={isTraining}
      >
        <img
          src="/person-chalkboard-solid-full.svg"
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
        <img src="/shuffle-solid-full.svg" alt="" width="16" height="16" />
        <span>Netz randomisieren</span>
      </button>

      <button
        class="btn-hover {isTraining ? 'btn-is-training' : ''}"
        onclick={handleTrainingButtonClick}
        disabled={busy}
      >
        {#if isTraining}
          <img src="/stop-solid-full.svg" alt="" width="16" height="16" />
          <span>Abbrechen</span>
        {:else}
          <img src="/play-solid-full.svg" alt="" width="16" height="16" />
          <span>Training starten</span>
        {/if}</button
      >
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

      <div class="epochs">
        Epochen: {trainingEpochsDone}
      </div>

      <div class="values">
        <div><h4>Fehlerwerte</h4></div>
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
    <h2>Live-Netzansicht (Klicken zum Bearbeiten)</h2>
    <p class="hint">
      Klicke auf Gewichtslabels, um einzelne Gewichte zu setzen. Klicke auf
      Knoten in Hidden/Output, um Bias zu aendern.
    </p>
    <div class="layer-controls">
      <div class="layer-buttons">
        <span>Hidden Layer:</span>

        <button
          class="btn-hover"
          onclick={removeHiddenLayer}
          disabled={isTraining || activeTab.layers.length <= 2}
        >
          -</button
        >
        <button class="btn-hover" onclick={addHiddenLayer} disabled={isTraining}
          >+</button
        >
      </div>
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
    </div>
    <div class="graph-scroll">
      <NetworkGraph
        {graph}
        {orderedConnections}
        {highlightedConnectionId}
        {activeTab}
        {setInputNeuronValue}
        {editWeight}
        {editBias}
      />
    </div>
  </section>

  <footer class="status">
    <p>{status}</p>
    {#if errorText}
      <p class="error">Fehler: {errorText}</p>
    {/if}
  </footer>

  {#if datasetModalOpen}
    <div class="modal-backdrop">
      <TrainingsModal
        {activeTab}
        {trainingWindowPosition}
        {trainingWindowSize}
        {setDatasetRowInput}
        {setDatasetRowOutput}
        {addDatasetRow}
        {removeDatasetRow}
        {exportDatasetCsv}
        {onDatasetFileSelected}
        {startTrainingWindowDrag}
        {startTrainingWindowResize}
        bind:datasetModalOpen
      />
    </div>
  {/if}

  {#if datasetImportPromptOpen}
    <div class="modal-backdrop">
      <div class="modal-window import-modal" role="dialog" aria-modal="true">
        <div class="modal-head">
          <div class="modal-title">CSV-Import</div>
          <button onclick={closeDatasetImportPrompt}>X</button>
        </div>

        <p>Hat die CSV einen Header?</p>
        <p class="csv-first-line">Erste Zeile: {pendingImportFirstLine}</p>

        <div class="import-actions">
          <button class="btn-hover" onclick={() => importDatasetCsv(true)}
            >Mit Header importieren</button
          >
          <button class="btn-hover" onclick={() => importDatasetCsv(false)}
            >Ohne Header importieren</button
          >
          <button class="btn-hover" onclick={closeDatasetImportPrompt}
            >Abbrechen</button
          >
        </div>
      </div>
    </div>
  {/if}
</main>

<style>
  .layer-buttons {
    display: flex;
    align-items: center;
    border-radius: 10px;
    height: 40px;
  }

  .layer-buttons > span {
    background-color: lightgray;
    height: 100%;
    align-content: center;
    padding-left: 5px;
    padding-right: 5px;
    border-top-left-radius: 10px;
    border-bottom-left-radius: 10px;
  }

  .layer-buttons > button {
    border-radius: 0;
    display: flex;
    align-items: center;
    height: 100%;
    font-size: 20px;
  }

  .layer-buttons > button:last-child {
    border-top-right-radius: 10px;
    border-bottom-right-radius: 10px;
  }
</style>
