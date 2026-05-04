/* global Go */

let runtimeReady = false;
let runtimeReadyPromise = null;
let trainerCounter = 0;
const trainers = new Map();
let runtimeBaseUrl = "/";

function normalizeBaseUrl(raw) {
  const value = String(raw || "/").trim();
  if (!value) {
    return "/";
  }
  const withLeadingSlash = value.startsWith("/") ? value : `/${value}`;
  return withLeadingSlash.endsWith("/")
    ? withLeadingSlash
    : `${withLeadingSlash}/`;
}

function resolveRuntimeAsset(fileName) {
  return `${runtimeBaseUrl}${fileName}`;
}

async function loadRuntimeScript(url) {
  if (typeof importScripts === "function") {
    importScripts(url);
    return;
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(
      `Script konnte nicht geladen werden: ${url} (HTTP ${response.status}).`,
    );
  }

  const code = await response.text();
  // Fallback fuer module worker ohne importScripts.
  globalThis.eval(code);
}

function makeTrainerId() {
  trainerCounter += 1;
  return `trainer-${Date.now()}-${trainerCounter}`;
}

async function waitForApi() {
  const timeoutAt = Date.now() + 5000;
  while (Date.now() < timeoutAt) {
    if (
      typeof self.nnCreateState === "function" &&
      typeof self.nnForward === "function" &&
      typeof self.nnTrain === "function" &&
      typeof self.nnListActivations === "function" &&
      typeof self.nnTrainerInit === "function" &&
      typeof self.nnTrainerStart === "function" &&
      typeof self.nnTrainerStatus === "function" &&
      typeof self.nnTrainerStop === "function" &&
      typeof self.nnTrainerDispose === "function"
    ) {
      return;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
  }
  throw new Error("WASM-API konnte im Worker nicht initialisiert werden.");
}

async function initRuntime() {
  if (runtimeReady) {
    return;
  }
  if (runtimeReadyPromise) {
    return runtimeReadyPromise;
  }

  runtimeReadyPromise = (async () => {
    const wasmExecUrl = resolveRuntimeAsset("wasm_exec.js");
    const wasmBinaryUrl = resolveRuntimeAsset("nn.wasm");
    await loadRuntimeScript(wasmExecUrl);

    if (typeof Go !== "function") {
      throw new Error("Go Runtime in wasm_exec.js nicht verfügbar.");
    }

    const go = new Go();
    let instance;

    try {
      const result = await WebAssembly.instantiateStreaming(
        fetch(wasmBinaryUrl),
        go.importObject,
      );
      instance = result.instance;
    } catch {
      const response = await fetch(wasmBinaryUrl);
      if (!response.ok) {
        throw new Error(
          `nn.wasm konnte nicht geladen werden (HTTP ${response.status}).`,
        );
      }
      const bytes = await response.arrayBuffer();
      const result = await WebAssembly.instantiate(bytes, go.importObject);
      instance = result.instance;
    }

    go.run(instance);
    await waitForApi();
    runtimeReady = true;
  })();

  return runtimeReadyPromise;
}

function callWasm(method, payload) {
  const fn = self[method];
  if (typeof fn !== "function") {
    throw new Error(`WASM-Methode nicht gefunden: ${method}`);
  }

  const raw = fn(JSON.stringify(payload));
  const parsed = JSON.parse(raw);

  if (parsed?.error) {
    throw new Error(parsed.error);
  }

  return parsed;
}

function getTrainer(trainerId) {
  const trainer = trainers.get(trainerId);
  if (!trainer) {
    throw new Error("trainer_id nicht gefunden");
  }
  return trainer;
}

function computeMetrics(state, dataset) {
  let maxDeviation = 0;
  let lossSum = 0;

  for (const sample of dataset) {
    const response = callWasm("nnForward", {
      state,
      input: sample.input,
    });

    for (let i = 0; i < sample.target.length; i += 1) {
      const outputValue = Number(response.output?.[i] ?? 0);
      const expectedValue = Number(sample.target[i]);
      const diff = outputValue - expectedValue;
      lossSum += 0.5 * diff * diff;

      const absDiff = Math.abs(diff);
      if (absDiff > maxDeviation) {
        maxDeviation = absDiff;
      }
    }
  }

  return {
    maxDeviation,
    loss: dataset.length > 0 ? lossSum / dataset.length : 0,
  };
}

function scheduleTrainerStep(trainerId) {
  const trainer = trainers.get(trainerId);
  if (!trainer || !trainer.running || trainer.scheduled) {
    return;
  }

  trainer.scheduled = true;
  setTimeout(() => runTrainerBatch(trainerId), 0);
}

function runTrainerBatch(trainerId) {
  const trainer = trainers.get(trainerId);
  if (!trainer) {
    return;
  }

  trainer.scheduled = false;

  if (!trainer.running || trainer.stopRequested) {
    trainer.running = false;
    return;
  }

  try {
    // CPU-Budget pro Batch: hoher Durchsatz, aber kooperativ fuer Status/Stop.
    const started = performance.now();
    const maxBatchMs = 25;
    const maxEpochsPerBatch = 32;
    let epochsInBatch = 0;

    while (trainer.running && !trainer.stopRequested) {
      const trainResult = callWasm("nnTrain", {
        state: trainer.state,
        dataset: trainer.dataset,
        epochs: 1,
        learning_rate: trainer.learningRate,
        shuffle: trainer.shuffle,
      });

      trainer.state = trainResult.state;
      trainer.epochsDone += 1;

      const metrics = computeMetrics(trainer.state, trainer.dataset);

      trainer.finalLoss = metrics.loss;
      trainer.hasFinalLoss = true;
      trainer.lossHistory.push(metrics.loss);

      trainer.deviation = metrics.maxDeviation;

      if (trainer.deviation === 0) {
        trainer.running = false;
        break;
      }

      epochsInBatch += 1;

      if (epochsInBatch >= maxEpochsPerBatch) {
        break;
      }
      if (performance.now() - started >= maxBatchMs) {
        break;
      }
    }
  } catch (error) {
    trainer.running = false;
    trainer.lastError = error instanceof Error ? error.message : String(error);
  }

  if (trainer.running && !trainer.stopRequested) {
    scheduleTrainerStep(trainerId);
  }
}

function handleTrainerMethod(method, payload) {
  if (method === "nnTrainerInit") {
    if (!Array.isArray(payload?.dataset) || payload.dataset.length === 0) {
      throw new Error("dataset darf nicht leer sein");
    }

    const trainerId = makeTrainerId();
    trainers.set(trainerId, {
      id: trainerId,
      state: payload.state,
      dataset: payload.dataset,
      learningRate: payload.learning_rate,
      shuffle: Boolean(payload.shuffle),
      running: false,
      stopRequested: false,
      scheduled: false,
      epochsDone: 0,
      lossHistory: [],
      finalLoss: 0,
      hasFinalLoss: false,
      deviation: 0,
      lastError: "",
    });
    return { trainer_id: trainerId };
  }

  if (method === "nnTrainerStart") {
    const trainer = getTrainer(payload?.trainer_id);
    trainer.running = true;
    trainer.stopRequested = false;
    trainer.lastError = "";
    scheduleTrainerStep(trainer.id);
    return { ok: true };
  }

  if (method === "nnTrainerStatus") {
    const trainer = getTrainer(payload?.trainer_id);
    if (trainer.lastError) {
      throw new Error(trainer.lastError);
    }
    return {
      trainer_id: trainer.id,
      running: trainer.running,
      epochs_done: trainer.epochsDone,
      loss_history: trainer.lossHistory,
      final_loss: trainer.finalLoss,
      has_final_loss: trainer.hasFinalLoss,
      deviation: trainer.deviation,
      state: trainer.state,
    };
  }

  if (method === "nnTrainerStop") {
    const trainer = getTrainer(payload?.trainer_id);
    trainer.stopRequested = true;
    trainer.running = false;
    return { ok: true };
  }

  if (method === "nnTrainerDispose") {
    trainers.delete(payload?.trainer_id);
    return { ok: true };
  }

  return null;
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (!msg || typeof msg !== "object") {
    return;
  }

  const { type, id, method, payload } = msg;

  try {
    if (type === "init") {
      runtimeBaseUrl = normalizeBaseUrl(msg.baseUrl);
      await initRuntime();
      self.postMessage({ type: "ready", id });
      return;
    }

    if (type === "call") {
      await initRuntime();
      const trainerResult = handleTrainerMethod(method, payload);
      const result =
        trainerResult === null ? callWasm(method, payload) : trainerResult;
      self.postMessage({ type: "response", id, ok: true, result });
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({ type: "response", id, ok: false, error: message });
  }
};
