<script>
  let { lossChart } = $props();
</script>

<svg
  viewBox={`0 0 ${lossChart.width} ${lossChart.height}`}
  class="loss-chart"
  role="img"
  aria-label="Loss Verlauf mit Skalen"
>
  <line
    class="loss-axis"
    x1={lossChart.yAxisX}
    y1={lossChart.yAxisTop}
    x2={lossChart.yAxisX}
    y2={lossChart.xAxisY}
  ></line>
  <line
    class="loss-axis"
    x1={lossChart.yAxisX}
    y1={lossChart.xAxisY}
    x2={lossChart.yAxisX + lossChart.plotWidth}
    y2={lossChart.xAxisY}
  ></line>

  {#each lossChart.yTicks as tick}
    <line
      class="loss-grid"
      x1={lossChart.yAxisX}
      y1={tick.y}
      x2={lossChart.yAxisX + lossChart.plotWidth}
      y2={tick.y}
    ></line>
    <text class="loss-tick loss-tick-y" x={lossChart.yAxisX - 6} y={tick.y + 3}
      >{tick.label}</text
    >
  {/each}

  {#each lossChart.xTicks as tick}
    <line
      class="loss-tick-mark"
      x1={tick.x}
      y1={lossChart.xAxisY}
      x2={tick.x}
      y2={lossChart.xAxisY + 4}
    ></line>
    <text class="loss-tick loss-tick-x" x={tick.x} y={lossChart.xAxisY + 16}
      >{tick.epoch}</text
    >
  {/each}

  {#if lossChart.hasLine}
    <polyline
      points={lossChart.linePoints}
      fill="none"
      stroke="var(--accent)"
      stroke-width="2.5"
    ></polyline>
  {/if}

  <text
    class="loss-axis-label"
    x={lossChart.yAxisX + lossChart.plotWidth / 2}
    y={lossChart.height - 4}
  >
    Epoche
  </text>
  <text class="loss-axis-label" x="55" y={lossChart.yAxisTop - 15}
    >Fehlerwert</text
  >
</svg>

<style>
  .loss-chart {
    width: 100%;
    min-height: 180px;
    border-radius: 10px;
    background: rgba(0, 109, 119, 0.06);
    border: 1px solid var(--line);
  }

  .loss-axis {
    stroke: #0f172a;
    stroke-width: 1;
  }

  .loss-grid {
    stroke: rgba(15, 23, 42, 0.12);
    stroke-width: 1;
  }

  .loss-tick-mark {
    stroke: #0f172a;
    stroke-width: 1;
  }

  .loss-tick {
    fill: #334155;
    font-size: 12px;
  }

  .loss-tick-y {
    text-anchor: end;
  }

  .loss-tick-x {
    text-anchor: middle;
  }

  .loss-axis-label {
    fill: #0f172a;
    font-size: 13px;
    font-weight: 600;
    text-anchor: middle;
  }
</style>
