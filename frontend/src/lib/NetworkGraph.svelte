<script>
  let {
    graph,
    highlightedConnectionId,
    orderedConnections,
    activeTab,
    setInputNeuronValue,
    editWeight,
    editBias,
  } = $props();
</script>

<svg
  class="network-graph"
  viewBox={`0 0 ${graph.width} ${graph.height}`}
  role="img"
  aria-label="Netzwerkvisualisierung"
>
  {#each orderedConnections as conn (conn.id)}
    <g>
      <line
        class="edge-hit"
        x1={conn.x1}
        y1={conn.y1}
        x2={conn.x2}
        y2={conn.y2}
        role="button"
        tabindex="0"
        onpointerenter={() => {
          highlightedConnectionId = conn.id;
        }}
        onpointerleave={() => {
          if (highlightedConnectionId === conn.id) {
            highlightedConnectionId = "";
          }
        }}
        onfocus={() => {
          highlightedConnectionId = conn.id;
        }}
        onclick={() => editWeight(conn)}
        onkeydown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            editWeight(conn);
          }
        }}
      ></line>
      <line
        class={`edge ${conn.weight >= 0 ? "pos" : "neg"}`}
        x1={conn.x1}
        y1={conn.y1}
        x2={conn.x2}
        y2={conn.y2}
        stroke-width={Math.min(3, 0.8 + Math.abs(conn.weight) * 0.45)}
      ></line>
      <text
        class={`edge-value ${highlightedConnectionId === conn.id ? "edge-value-active" : ""}`}
        x={conn.labelX}
        y={conn.labelY}
        role="button"
        tabindex="0"
        onpointerenter={() => {
          highlightedConnectionId = conn.id;
        }}
        onpointerleave={() => {
          if (highlightedConnectionId === conn.id) {
            highlightedConnectionId = "";
          }
        }}
        onclick={() => editWeight(conn)}
        onkeydown={(e) => {
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
        <circle class="node input" cx={node.x} cy={node.y} r="16"></circle>
        <foreignObject
          class="node-input-wrap"
          x={Math.max(8, node.x - 70)}
          y={node.y - 13}
          width="50"
          height="22"
        >
          <input
            class="node-input"
            type="number"
            value={activeTab.inputNeuronValues?.[node.node] ?? "0"}
            oninput={(e) =>
              setInputNeuronValue(node.node, e.currentTarget.value)}
          />
        </foreignObject>
      {:else}
        <circle
          class="node editable"
          cx={node.x}
          cy={node.y}
          r="16"
          role="button"
          tabindex="0"
          onclick={() => editBias(node.layer, node.node)}
          onkeydown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              editBias(node.layer, node.node);
            }
          }}
        ></circle>
      {/if}
      <text class="node-index" x={node.x} y={node.y + 4}>N{node.node + 1}</text>
      {#if node.layer > 0 && activeTab.state}
        <text class="node-bias" x={node.x} y={node.y + 28}>
          b:{Number(activeTab.state.biases[node.layer - 1][node.node]).toFixed(
            2,
          )}
        </text>
      {/if}
      {#if node.layer === activeTab.layers.length - 1}
        <text class="node-output" x={node.x + 24} y={node.y + 4}>
          {activeTab.outputNeuronValues?.[node.node] ?? "-"}
        </text>
      {/if}
    </g>
  {/each}
</svg>

<style>
  .edge {
    opacity: 0.65;
    pointer-events: none;
  }

  .edge-hit {
    stroke: transparent;
    stroke-width: 8;
    pointer-events: stroke;
    cursor: pointer;
  }

  .edge.pos {
    stroke: #1f8f6a;
  }

  .edge.neg {
    stroke: #bf5d43;
  }

  .edge-value {
    font-size: 10px;
    fill: #111827;
    paint-order: stroke;
    stroke: #f8fafc;
    stroke-width: 2px;
    cursor: pointer;
  }

  .edge-value-active {
    fill: #ffffff;
    font-weight: 700;
    stroke: #1d1d1d;
    stroke-width: 3px;
  }

  .node {
    stroke-width: 2;
  }

  .node.input {
    fill: #dbeafe;
    stroke: #3b82f6;
  }

  .node.editable {
    fill: #dcfce7;
    stroke: #22a362;
    cursor: pointer;
  }

  .node-index {
    text-anchor: middle;
    font-size: 10px;
    fill: #111827;
  }

  .node-bias {
    text-anchor: middle;
    font-size: 9px;
    fill: #475569;
  }

  .node-input-wrap {
    overflow: visible;
  }

  .node-input {
    width: 100%;
    height: 100%;
    border: 1px solid #93b2eb;
    border-radius: 6px;
    background: rgba(248, 251, 255, 0.97);
    text-align: center;
    padding: 0.08rem 0.2rem;
    font-size: 11px;
  }

  .node-input::-webkit-outer-spin-button,
  .node-input::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0; /* <-- Apparently some margin are still there even though it's hidden */
  }

  .node-input[type="number"] {
    appearance: textfield; /* Safari, Chrome, Edge */
    -moz-appearance: textfield; /* Firefox */
  }

  .node-output {
    font-size: 10px;
    fill: #0f766e;
    font-weight: 600;
  }
</style>
