<script>
  import SevenSegment from "./SevenSegment.svelte";

  const publicAsset = (fileName) => `${import.meta.env.BASE_URL}${fileName}`;

  let {
    activeTab,
    trainingWindowPosition,
    trainingWindowSize,
    datasetImportPromptOpen,
    datasetImportStep,
    pendingImportFirstLine,
    pendingImportSecondLine,
    setDatasetRowInput,
    setDatasetRowOutput,
    editInputNeuronName,
    editOutputNeuronName,
    addDatasetRow,
    removeDatasetRow,
    exportDatasetCsv,
    onDatasetFileSelected,
    answerImportSecondLineIsNames,
    answerImportAdoptNames,
    closeDatasetImportPrompt,
    startTrainingWindowDrag,
    startTrainingWindowResize,
    datasetModalOpen = $bindable(true),
    trainingImportError = $bindable(""),
    showOutputSegment,
  } = $props();

  function getOutputIndexByName(neuronName) {
    return activeTab.outputNeuronNames?.findIndex(
      (name) => String(name).toLowerCase() === neuronName,
    );
  }

  function getRowSegmentValue(rowIndex, neuronName) {
    const neuronIndex = getOutputIndexByName(neuronName);
    if (neuronIndex === undefined || neuronIndex < 0) {
      return false;
    }

    const row = activeTab.datasetRows?.[rowIndex];
    const value = Number(row?.target?.[neuronIndex]);
    return Number.isFinite(value) && value >= 0.5;
  }

  function setRowSegmentValue(rowIndex, neuronName, newValue) {
    const neuronIndex = getOutputIndexByName(neuronName);
    if (neuronIndex === undefined || neuronIndex < 0) {
      return;
    }

    setDatasetRowOutput(rowIndex, neuronIndex, newValue ? 1 : 0);
  }

  function createRowSegmentBinding(rowIndex) {
    const binding = {};

    for (const name of ["a", "b", "c", "d", "e", "f", "g"]) {
      Object.defineProperty(binding, name, {
        enumerable: true,
        get() {
          return getRowSegmentValue(rowIndex, name);
        },
        set(nextValue) {
          setRowSegmentValue(rowIndex, name, Boolean(nextValue));
        },
      });
    }

    return binding;
  }

  let segmentBindings = $derived.by(() =>
    (activeTab.datasetRows || []).map((_, rowIndex) =>
      createRowSegmentBinding(rowIndex),
    ),
  );
</script>

<div
  class="modal-window dataset-modal resizable-window"
  role="dialog"
  aria-modal="true"
  style={`left:${trainingWindowPosition.x}px;top:${trainingWindowPosition.y}px;width:${trainingWindowSize.width}px;height:${trainingWindowSize.height}px;`}
>
  <div
    class="modal-head modal-drag-handle"
    role="button"
    tabindex="0"
    onmousedown={startTrainingWindowDrag}
  >
    <div class="modal-title">Trainingsdaten bearbeiten</div>
    <button class="btn-hover" onclick={() => (datasetModalOpen = false)}
      >X</button
    >
  </div>

  <button
    type="button"
    class="resize-handle resize-se"
    aria-label="Fenster unten rechts skalieren"
    onmousedown={(e) => startTrainingWindowResize(e, "se")}
  ></button>

  <div class="dataset-actions">
    <div class="button-group">
      <span>Trainingsdaten:</span>

      <button class="btn-hover" onclick={exportDatasetCsv}>
        <img
          src={publicAsset("floppy-disk-solid-full.svg")}
          alt=""
          width="20"
          height="20"
        />
        Speichern</button
      >
      <label class="btn-file btn-hover">
        <img
          src={publicAsset("upload-solid-full.svg")}
          alt=""
          width="20"
          height="20"
        />
        Öffnen
        <input
          type="file"
          accept=".csv,text/csv"
          onchange={onDatasetFileSelected}
        />
      </label>
    </div>
  </div>

  {#if trainingImportError !== ""}
    <div class="inline-import-box">
      <p class="inline-import-line text-danger">{@html trainingImportError}</p>
      <div class="inline-import-actions">
        <button class="btn-hover" onclick={() => (trainingImportError = "")}
          >OK</button
        >
      </div>
    </div>
  {/if}

  {#if datasetImportPromptOpen}
    <div class="inline-import-box">
      <div class="inline-import-title">CSV-Import prüfen:</div>
      <p class="inline-import-line">
        Erste Zeile: <code>{pendingImportFirstLine}</code>
      </p>
      <p class="inline-import-line">
        Zweite Zeile: <code>{pendingImportSecondLine}</code>
      </p>

      {#if datasetImportStep === "ask-second-line"}
        <p class="inline-import-question">
          Sind die Werte in der zweiten Zeile die Namen der Neuronen?
        </p>
        <div class="inline-import-actions">
          <button class="btn-hover" onclick={() => answerImportAdoptNames(true)}
            >Ja, und die Namen übernehmen</button
          >
          <button
            class="btn-hover"
            onclick={() => answerImportAdoptNames(false)}
            >Ja, aber die aktuellen Namen beibehalten</button
          >
          <button
            class="btn-hover"
            onclick={() => answerImportSecondLineIsNames(false)}
            >Nein, keine Namen</button
          >
          <button class="btn-hover" onclick={closeDatasetImportPrompt}
            >Import abbrechen</button
          >
        </div>
      {/if}
    </div>
  {/if}

  <div class="dataset-grid-wrap">
    <table class="dataset-grid">
      <thead>
        <tr>
          <th colspan={activeTab.layers[0] + 1}>Inputs</th>
          <th
            colspan={activeTab.layers[activeTab.layers.length - 1]}
            class="outputs-header">Outputs</th
          >
          <th></th>
          {#if showOutputSegment}
            <th></th>
          {/if}
        </tr>
        <tr>
          <th>Nr.</th>
          {#each Array.from({ length: activeTab.layers[0] }, (_, idx) => idx) as idx}
            <th>
              <button
                type="button"
                class="input-name-btn"
                title="Input-Namen ändern"
                onclick={() => editInputNeuronName(idx)}
              >
                {activeTab.inputNeuronNames?.[idx] ?? `input${idx + 1}`}
              </button>
            </th>
          {/each}
          {#each Array.from({ length: activeTab.layers[activeTab.layers.length - 1] }, (_, idx) => idx) as idx}
            <th class={idx === 0 ? "outputs-header" : ""}>
              <button
                type="button"
                class="input-name-btn"
                title="Output-Namen ändern"
                onclick={() => editOutputNeuronName(idx)}
              >
                {activeTab.outputNeuronNames?.[idx] ?? `output${idx + 1}`}
              </button>
            </th>
          {/each}
          <th></th>
          {#if showOutputSegment}
            <th></th>
          {/if}
        </tr>
      </thead>
      <tbody>
        {#each activeTab.datasetRows as row, rowIndex}
          <tr>
            <td>{rowIndex + 1}</td>
            {#each row.input as value, inputIndex}
              <td>
                <input
                  type="number"
                  {value}
                  oninput={(e) =>
                    setDatasetRowInput(
                      rowIndex,
                      inputIndex,
                      e.currentTarget.value,
                    )}
                />
              </td>
            {/each}
            {#each row.target as value, outputIndex}
              <td class={outputIndex === 0 ? "outputs-header" : ""}>
                <input
                  type="number"
                  {value}
                  oninput={(e) =>
                    setDatasetRowOutput(
                      rowIndex,
                      outputIndex,
                      e.currentTarget.value,
                    )}
                />
              </td>
            {/each}
            {#if showOutputSegment}
              <td class="segment-cell">
                <SevenSegment
                  bind:a={segmentBindings[rowIndex].a}
                  bind:b={segmentBindings[rowIndex].b}
                  bind:c={segmentBindings[rowIndex].c}
                  bind:d={segmentBindings[rowIndex].d}
                  bind:e={segmentBindings[rowIndex].e}
                  bind:f={segmentBindings[rowIndex].f}
                  bind:g={segmentBindings[rowIndex].g}
                  editable={true}
                />
              </td>
            {/if}
            <td>
              <button
                class="btn-hover"
                onclick={() => removeDatasetRow(rowIndex)}
                disabled={activeTab.datasetRows.length <= 1}
                tabIndex="-1"
                ><img
                  src={publicAsset("trash-solid-full.svg")}
                  alt=""
                  width="16"
                  height="16"
                /></button
              >
            </td>

            <!-- //activeTab.outputNeuronNames?.find((n, i) => {if (n === "a") { row.target[i] >= 0.5} else false} -->
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
  <div class="addRow">
    <button class="btn-hover" onclick={addDatasetRow}>+ Zeile</button>
  </div>
</div>

<style>
  .segment-cell {
    min-width: 55px;
    width: 55px;
    height: 80px;
  }

  .button-group > * {
    font-size: inherit;
  }

  .btn-file {
    padding-top: 0;
    padding-bottom: 0;
    border-radius: 0;
  }

  tbody > tr:nth-child(odd) {
    background: rgba(0, 0, 0, 0.03);
  }

  .addRow {
    display: flex;
    justify-content: center;
    padding: 0.5rem 0;
  }

  :global(code) {
    background-color: rgb(66, 66, 66);
    padding: 0.2rem;
    color: white;
    border-radius: 4px;
  }

  .text-danger {
    color: var(--danger);
  }

  th {
    font-family: sans-serif;
  }

  .dataset-modal {
    min-width: 540px;
    min-height: 320px;
    max-width: calc(100vw - 20px);
    max-height: calc(100vh - 20px);
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }

  .modal-head {
    flex: 0 0 auto;
  }

  .outputs-header {
    border-left: 1px solid var(--line);
  }

  .dataset-actions {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-bottom: 0.7rem;
    flex: 0 0 auto;
  }

  .dataset-grid-wrap {
    flex: 1 1 auto;
    min-height: 0;
    overflow: auto;
    border: 1px solid var(--line);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.65);
    align-self: center;
  }

  .dataset-grid {
    width: 100%;
    border-collapse: collapse;
  }

  .dataset-grid th,
  .dataset-grid td {
    border-bottom: 1px solid var(--line);
    padding: 0.35rem;
    text-align: center;
  }

  .dataset-grid th {
    font-size: 0.78rem;
    background: rgba(0, 0, 0, 0.03);
  }

  .input-name-btn {
    border: 0;
    background: transparent;
    color: inherit;
    font: inherit;
    font-weight: 600;
    text-decoration: underline;
    text-decoration-thickness: 1px;
    text-underline-offset: 2px;
    cursor: pointer;
    padding: 0;
  }

  .dataset-grid td input {
    width: 3rem;
  }

  .inline-import-box {
    border: 1px solid var(--line);
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.85);
    padding: 0.6rem 0.7rem;
    margin-bottom: 0.7rem;
  }

  .inline-import-title {
    font-weight: 700;
    margin-bottom: 0.35rem;
  }

  .inline-import-line,
  .inline-import-question {
    margin: 0.2rem 0;
    font-size: 0.82rem;
  }

  .inline-import-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.45rem;
  }

  .resize-handle {
    position: absolute;
    right: 0px;
    bottom: 0px;
    z-index: 2;
    border: 0;
    border-radius: 5px;
    margin: 0;
    padding: 0;
    background: lightgrey;
  }

  .resize-se {
    width: 18px;
    height: 18px;
    cursor: nwse-resize;
  }
</style>
