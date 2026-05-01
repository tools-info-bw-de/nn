<script>
  import { onMount } from "svelte";

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

  const defaultInput = "1, 0";

  let wasmReady = false;
  let busy = false;
  let status = "Initialisiere...";
  let errorText = "";

  let availableActivations = ["binary", "logistic", "relu"];

  let tabCounter = 1;
  let tabs = [createTab(tabCounter)];
  let activeTabId = tabs[0].id;

  let renamingTabId = "";
  let renameDraft = "";

  let lossWindowOpen = false;
  let lossWindowPosition = { x: 110, y: 90 };
  let lossWindowDragging = false;
  let lossWindowDragOffset = { x: 0, y: 0 };

  let isTraining = false;
  let stopTrainingRequested = false;
  let trainingTabId = "";
  let trainingEpochsDone = 0;
  let trainingLastLoss = null;
  let trainingDeviation = null;

  function clone(value) {
    return JSON.parse(JSON.stringify(value));
  }

  function createTab(nr) {
    return {
      id: `tab-${Date.now()}-${Math.random().toString(36).slice(2)}`,
      name: `Netz ${nr}`,
      layers: [2, 3, 1],
      activation: "logistic",
      learningRate: 0.1,
      epochs: 200,
      shuffle: true,
      datasetText: defaultDataset,
      forwardInputText: defaultInput,
      forwardOutputText: "",
      lossHistory: [],
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

  function parseNumberList(text) {
    return text
      .split(",")
      .map((value) => Number(value.trim()))
      .filter((value) => Number.isFinite(value));
  }

  function parseJsonResponse(raw) {
    const parsed = JSON.parse(raw);
    if (parsed?.error) {
      throw new Error(parsed.error);
    }
    return parsed;
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
      const response = parseJsonResponse(
        window.nnForward(
          JSON.stringify({
            state,
            input: sample.input,
          }),
        ),
      );

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

    if (typeof window.Go !== "function") {
      throw new Error(
        "wasm_exec.js wurde nicht geladen. Stelle sicher, dass public/wasm_exec.js vorhanden ist.",
      );
    }

    const go = new window.Go();
    let instance;

    try {
      const result = await WebAssembly.instantiateStreaming(
        fetch("/nn.wasm"),
        go.importObject,
      );
      instance = result.instance;
    } catch {
      const response = await fetch("/nn.wasm");
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
    wasmReady = true;

    const listed = parseJsonResponse(window.nnListActivations());
    if (Array.isArray(listed?.activations) && listed.activations.length > 0) {
      availableActivations = listed.activations;
    }
  }

  function requireApi() {
    const ok =
      typeof window.nnCreateState === "function" &&
      typeof window.nnTrain === "function" &&
      typeof window.nnForward === "function" &&
      typeof window.nnListActivations === "function";

    if (!ok) {
      throw new Error(
        "WASM-API nicht verfuegbar. Build der Go-WASM-Datei pruefen.",
      );
    }
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
    requireApi();

    const tab = tabs.find((item) => item.id === tabId);
    if (!tab) {
      throw new Error("Tab nicht gefunden.");
    }

    const payload = {
      layers: tab.layers,
      activations: buildActivationList(tab.layers, tab.activation),
      learning_rate: Number(tab.learningRate),
    };

    const result = parseJsonResponse(
      window.nnCreateState(JSON.stringify(payload)),
    );

    updateTab(tabId, (t) => {
      t.state = result.state;
      t.forwardOutputText = "";
      t.lossHistory = [];
    });
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

  async function trainActive() {
    if (isTraining) {
      return;
    }

    errorText = "";

    try {
      await initWasm();
      requireApi();
      await ensureStateForActiveTab();

      const active = getActiveTab();
      const dataset = JSON.parse(active.datasetText);
      if (!Array.isArray(dataset) || dataset.length === 0) {
        throw new Error(
          "Trainingsdaten muessen ein nicht-leeres JSON-Array sein.",
        );
      }

      isTraining = true;
      stopTrainingRequested = false;
      trainingTabId = active.id;
      trainingEpochsDone = 0;
      trainingLastLoss = null;
      trainingDeviation = null;
      lossWindowOpen = true;
      status = `${active.name}: Training gestartet.`;

      while (!stopTrainingRequested) {
        const tab = tabs.find((entry) => entry.id === trainingTabId);
        if (!tab) {
          status = "Training gestoppt: Trainings-Tab wurde geschlossen.";
          break;
        }

        const payload = {
          state: tab.state,
          dataset,
          epochs: 1,
          learning_rate: Number(tab.learningRate),
          shuffle: Boolean(tab.shuffle),
        };

        const result = parseJsonResponse(
          window.nnTrain(JSON.stringify(payload)),
        );
        const currentLoss = Number(result.final_loss);
        const currentDeviation = await computeMaxDeviation(
          result.state,
          dataset,
        );

        updateTab(trainingTabId, (next) => {
          next.state = result.state;
          next.lossHistory = [...next.lossHistory, currentLoss];
        });

        trainingEpochsDone += 1;
        trainingLastLoss = currentLoss;
        trainingDeviation = currentDeviation;

        status = `${tab.name}: Training Epoche ${trainingEpochsDone}, Loss ${currentLoss.toFixed(6)}, Abweichung ${currentDeviation.toFixed(6)}`;

        if (currentDeviation === 0) {
          status = `${tab.name}: Ziel erreicht (Abweichung 0 nach ${trainingEpochsDone} Epochen).`;
          break;
        }

        await delay(50);
      }

      if (stopTrainingRequested) {
        const tab = tabs.find((entry) => entry.id === trainingTabId);
        status = `${tab?.name ?? "Netz"}: Training manuell abgebrochen.`;
      }
    } catch (error) {
      errorText = error instanceof Error ? error.message : String(error);
    } finally {
      isTraining = false;
      stopTrainingRequested = false;
      trainingTabId = "";
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

  function inferActive() {
    return withBusy(async () => {
      await initWasm();
      requireApi();
      await ensureStateForActiveTab();

      const active = getActiveTab();
      const input = parseNumberList(active.forwardInputText);
      const payload = {
        state: active.state,
        input,
      };

      const result = parseJsonResponse(
        window.nnForward(JSON.stringify(payload)),
      );
      updateActiveTab((t) => {
        t.forwardOutputText = JSON.stringify(result.output);
      });

      status = `${active.name}: Inferenz ausgefuehrt.`;
    });
  }

  function addTab() {
    tabCounter += 1;
    const next = createTab(tabCounter);
    tabs = [...tabs, next];
    activeTabId = next.id;
    status = `${next.name} erstellt.`;
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
      tab.forwardOutputText = "";
    });
  }

  function addHiddenLayer() {
    updateActiveTab((tab) => {
      tab.layers.splice(tab.layers.length - 1, 0, 3);
      tab.state = null;
      tab.lossHistory = [];
      tab.forwardOutputText = "";
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
      tab.forwardOutputText = "";
    });
  }

  function setDatasetText(value) {
    updateActiveTab((tab) => {
      tab.datasetText = value;
    });
  }

  function setForwardInputText(value) {
    updateActiveTab((tab) => {
      tab.forwardInputText = value;
    });
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

  $: activeTab = tabs.find((tab) => tab.id === activeTabId) || tabs[0];
  $: stateForDraw = activeTab?.state || buildPlaceholderState(activeTab);
  $: graph = geometryFromState(stateForDraw);
  $: hasLoss = activeTab?.lossHistory?.length > 0;

  function startLossWindowDrag(event) {
    if (event.button !== 0) {
      return;
    }

    lossWindowDragging = true;
    lossWindowDragOffset = {
      x: event.clientX - lossWindowPosition.x,
      y: event.clientY - lossWindowPosition.y,
    };
  }

  function onGlobalMouseMove(event) {
    if (!lossWindowDragging) {
      return;
    }

    const nextX = event.clientX - lossWindowDragOffset.x;
    const nextY = event.clientY - lossWindowDragOffset.y;
    const maxX = Math.max(10, window.innerWidth - 280);
    const maxY = Math.max(10, window.innerHeight - 180);

    lossWindowPosition = {
      x: Math.min(Math.max(10, nextX), maxX),
      y: Math.min(Math.max(10, nextY), maxY),
    };
  }

  function onGlobalMouseUp() {
    lossWindowDragging = false;
  }

  $: lossPath = (() => {
    const history = activeTab?.lossHistory || [];
    if (history.length < 2) {
      return "";
    }

    const w = 640;
    const h = 220;
    const min = Math.min(...history);
    const max = Math.max(...history);
    const range = max - min || 1;

    return history
      .map((v, i) => {
        const x = (i / (history.length - 1)) * w;
        const y = h - ((v - min) / range) * h;
        return `${x},${y}`;
      })
      .join(" ");
  })();

  onMount(() => {
    window.addEventListener("mousemove", onGlobalMouseMove);
    window.addEventListener("mouseup", onGlobalMouseUp);

    withBusy(async () => {
      await initWasm();
      requireApi();
      await createStateForTab(activeTabId);
      status = "Bereit. Du kannst jetzt pro Tab ein separates Netz bearbeiten.";
    });

    return () => {
      window.removeEventListener("mousemove", onGlobalMouseMove);
      window.removeEventListener("mouseup", onGlobalMouseUp);
    };
  });
</script>

<main class="app-shell">
  <header class="tabs-header">
    <div class="tab-row">
      {#each tabs as tab}
        <div class={`tab-pill ${tab.id === activeTabId ? "active" : ""}`}>
          <button class="tab-open" on:click={() => (activeTabId = tab.id)}>
            {#if renamingTabId === tab.id}
              <input
                class="rename-input"
                bind:value={renameDraft}
                on:blur={finishRename}
                on:keydown={(e) => {
                  if (e.key === "Enter") {
                    finishRename();
                  }
                  if (e.key === "Escape") {
                    renamingTabId = "";
                  }
                }}
              />
            {:else}
              <span
                role="button"
                tabindex="0"
                on:click|stopPropagation={() => beginRename(tab)}
                on:keydown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    beginRename(tab);
                  }
                }}
              >
                {tab.name}
              </span>
            {/if}
          </button>
          <button class="tab-edit" on:click={() => beginRename(tab)}>✎</button>
          <button
            class="tab-close"
            on:click={() => closeTab(tab.id)}
            disabled={tabs.length === 1}>×</button
          >
        </div>
      {/each}
      <button class="tab-add" on:click={addTab}>+ Tab</button>
    </div>
  </header>

  <section class="toolbar">
    <label>
      Aktivierung
      <select
        value={activeTab.activation}
        on:change={(e) => setActiveActivation(e.currentTarget.value)}
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
        min="0.000001"
        step="0.001"
        value={activeTab.learningRate}
        on:input={(e) => setActiveLearningRate(e.currentTarget.value)}
      />
    </label>

    <button on:click={randomizeActiveState} disabled={busy || isTraining}
      >Netz randomisieren</button
    >

    <label class="dataset-field">
      Trainingsdaten (JSON)
      <textarea
        rows="6"
        value={activeTab.datasetText}
        disabled={isTraining}
        on:input={(e) => setDatasetText(e.currentTarget.value)}
      ></textarea>
    </label>

    <button on:click={handleTrainingButtonClick} disabled={busy}
      >{isTraining ? "Abbrechen" : "Training starten"}</button
    >
    <button on:click={() => (lossWindowOpen = true)} disabled={!hasLoss}
      >Fehlerentwicklung</button
    >
  </section>

  <section class="network-controls">
    <h2>Netzstruktur</h2>
    <div class="layer-controls">
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
            on:input={(e) => setLayerCount(layerIndex, e.currentTarget.value)}
          />
        </label>
      {/each}
      <div class="layer-buttons">
        <button on:click={addHiddenLayer} disabled={isTraining}
          >Hidden Layer +</button
        >
        <button
          on:click={removeHiddenLayer}
          disabled={isTraining || activeTab.layers.length <= 2}
          >Hidden Layer -</button
        >
      </div>
    </div>
  </section>

  <section class="network-graph-wrap">
    <h2>Live-Netzansicht (Klicken zum Bearbeiten)</h2>
    <p class="hint">
      Klicke auf Gewichtslabels, um einzelne Gewichte zu setzen. Klicke auf
      Knoten in Hidden/Output, um Bias zu aendern.
    </p>
    <div class="graph-scroll">
      <svg
        class="network-graph"
        viewBox={`0 0 ${graph.width} ${graph.height}`}
        role="img"
        aria-label="Netzwerkvisualisierung"
      >
        {#each graph.connections as conn}
          <g>
            <line
              class={`edge ${conn.weight >= 0 ? "pos" : "neg"}`}
              x1={conn.x1}
              y1={conn.y1}
              x2={conn.x2}
              y2={conn.y2}
              stroke-width={Math.min(3, 0.8 + Math.abs(conn.weight) * 0.45)}
            ></line>
            <text
              class="edge-value"
              x={(conn.x1 + conn.x2) / 2}
              y={(conn.y1 + conn.y2) / 2}
              role="button"
              tabindex="0"
              on:click={() => editWeight(conn)}
              on:keydown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  editWeight(conn);
                }
              }}
            >
              {Number(conn.weight).toFixed(2)}
            </text>
          </g>
        {/each}

        {#each graph.nodes as node}
          <g>
            {#if node.layer === 0}
              <circle class="node input" cx={node.x} cy={node.y} r="16"
              ></circle>
            {:else}
              <circle
                class="node editable"
                cx={node.x}
                cy={node.y}
                r="16"
                role="button"
                tabindex="0"
                on:click={() => editBias(node.layer, node.node)}
                on:keydown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    editBias(node.layer, node.node);
                  }
                }}
              ></circle>
            {/if}
            <text class="node-index" x={node.x} y={node.y + 4}
              >N{node.node + 1}</text
            >
            {#if node.layer > 0 && activeTab.state}
              <text class="node-bias" x={node.x} y={node.y + 28}>
                b:{Number(
                  activeTab.state.biases[node.layer - 1][node.node],
                ).toFixed(2)}
              </text>
            {/if}
          </g>
        {/each}
      </svg>
    </div>
  </section>

  <section class="inference">
    <h2>Inferenz</h2>
    <label>
      Input (CSV)
      <input
        value={activeTab.forwardInputText}
        on:input={(e) => setForwardInputText(e.currentTarget.value)}
      />
    </label>
    <button on:click={inferActive} disabled={busy || isTraining}
      >Inferenz ausfuehren</button
    >
    <p class="output">Output: {activeTab.forwardOutputText || "-"}</p>
  </section>

  <footer class="status">
    <p>{status}</p>
    {#if errorText}
      <p class="error">Fehler: {errorText}</p>
    {/if}
  </footer>

  {#if lossWindowOpen}
    <div class="modal-backdrop">
      <div
        class="modal-window"
        role="dialog"
        aria-modal="true"
        style={`left: ${lossWindowPosition.x}px; top: ${lossWindowPosition.y}px;`}
      >
        <div class="modal-head">
          <div
            class="modal-drag-handle"
            role="button"
            tabindex="0"
            on:mousedown={startLossWindowDrag}
            on:keydown={(e) => {
              if (e.key === "Escape") {
                lossWindowOpen = false;
              }
            }}
          >
            Fehlerentwicklung pro Epoche
          </div>
          <button on:click={() => (lossWindowOpen = false)}>X</button>
        </div>

        {#if isTraining}
          <p class="loss-meta">
            Training laeuft. Epochen: {trainingEpochsDone} | Letzter Loss: {trainingLastLoss ===
            null
              ? "-"
              : trainingLastLoss.toFixed(6)} | Aktuelle Abweichung: {trainingDeviation ===
            null
              ? "-"
              : trainingDeviation.toFixed(6)}
          </p>
        {/if}

        {#if hasLoss}
          <svg
            viewBox="0 0 640 220"
            class="loss-chart"
            role="img"
            aria-label="Loss Verlauf"
          >
            <polyline
              points={lossPath}
              fill="none"
              stroke="var(--accent)"
              stroke-width="2.5"
            ></polyline>
          </svg>
          <p class="loss-meta">
            Min: {Math.min(...activeTab.lossHistory).toFixed(6)} | Max: {Math.max(
              ...activeTab.lossHistory,
            ).toFixed(6)} | Letzt: {activeTab.lossHistory[
              activeTab.lossHistory.length - 1
            ].toFixed(6)}
          </p>
        {:else}
          <p>Kein Verlauf vorhanden. Starte zuerst ein Training.</p>
        {/if}
      </div>
    </div>
  {/if}
</main>
