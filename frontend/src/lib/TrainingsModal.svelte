<script>
  let {
    activeTab,
    trainingWindowPosition,
    trainingWindowSize,
    setDatasetRowInput,
    setDatasetRowOutput,
    addDatasetRow,
    removeDatasetRow,
    exportDatasetCsv,
    onDatasetFileSelected,
    startTrainingWindowDrag,
    startTrainingWindowResize,
    datasetModalOpen = $bindable(true),
  } = $props();
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
    <button onclick={() => (datasetModalOpen = false)}>X</button>
  </div>

  <button
    type="button"
    class="resize-handle resize-se"
    aria-label="Fenster unten rechts skalieren"
    onmousedown={(e) => startTrainingWindowResize(e, "se")}
  ></button>

  <div class="dataset-actions">
    <button class="btn-hover" onclick={addDatasetRow}>+ Zeile</button>
    <button class="btn-hover" onclick={exportDatasetCsv}>CSV speichern</button>
    <label class="btn-file btn-hover">
      CSV laden
      <input
        type="file"
        accept=".csv,text/csv"
        onchange={onDatasetFileSelected}
      />
    </label>
  </div>

  <div class="dataset-grid-wrap">
    <table class="dataset-grid">
      <thead>
        <tr>
          <th colspan={activeTab.layers[0]}>Inputs</th>
          <th
            colspan={activeTab.layers[activeTab.layers.length - 1]}
            class="outputs-header">Outputs</th
          >
          <th>Aktion</th>
        </tr>
        <tr>
          {#each Array.from({ length: activeTab.layers[0] }, (_, idx) => idx) as idx}
            <th>I{idx + 1}</th>
          {/each}
          {#each Array.from({ length: activeTab.layers[activeTab.layers.length - 1] }, (_, idx) => idx) as idx}
            <th class={idx === 0 ? "outputs-header" : ""}>O{idx + 1}</th>
          {/each}
          <th></th>
        </tr>
      </thead>
      <tbody>
        {#each activeTab.datasetRows as row, rowIndex}
          <tr>
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
            <td>
              <button
                class="btn-hover"
                onclick={() => removeDatasetRow(rowIndex)}
                disabled={activeTab.datasetRows.length <= 1}>Loeschen</button
              >
            </td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</div>

<style>
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
  }

  .dataset-grid {
    width: 100%;
    border-collapse: collapse;
    min-width: 700px;
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

  .dataset-grid td input {
    width: 5.5rem;
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
