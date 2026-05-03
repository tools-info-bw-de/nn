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
