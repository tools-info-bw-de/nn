<script>
  // @ts-nocheck

  import { onMount } from "svelte";

  let {
    a = $bindable(false),
    b = $bindable(false),
    c = $bindable(false),
    d = $bindable(false),
    e = $bindable(false),
    f = $bindable(false),
    g = $bindable(false),
    editable = false,
  } = $props();

  let sevenSegment;
  let thickness = $state(0);
  let middleTop = $state(0);
  let horizontalWidth = $state(0);
  let verticalHeight = $state(0);

  function recalcGeometry() {
    if (!sevenSegment) return;

    const height = sevenSegment.clientHeight;
    const width = sevenSegment.clientWidth;

    thickness = Math.max(1, height * 0.1);
    horizontalWidth = Math.max(0, width - 2 * thickness);
    verticalHeight = Math.max(0, 0.5 * height - 1.5 * thickness);
    middleTop = Math.max(0, (height - thickness) / 2);
  }

  onMount(() => {
    recalcGeometry();

    const observer = new ResizeObserver(() => {
      recalcGeometry();
    });

    if (sevenSegment) observer.observe(sevenSegment);

    return () => {
      observer.disconnect();
    };
  });

  function clicked(segment) {
    if (!editable) return;

    switch (segment) {
      case "a":
        a = !a;
        break;
      case "b":
        b = !b;
        break;
      case "c":
        c = !c;
        break;
      case "d":
        d = !d;
        break;
      case "e":
        e = !e;
        break;
      case "f":
        f = !f;
        break;
      case "g":
        g = !g;
        break;
    }
  }
</script>

<!-- svelte-ignore a11y_click_events_have_key_events -->
<!-- svelte-ignore a11y_no_static_element_interactions -->
<div class="outer">
  <div bind:this={sevenSegment} class="seven-segment">
    <div
      class="segment a"
      style:cursor={editable ? "pointer" : "default"}
      onclick={() => {
        clicked("a");
      }}
      class:active={a}
      style:left={thickness + "px"}
      style:width={horizontalWidth + "px"}
      style:height={thickness + "px"}
    >
      a
    </div>
    <div
      class="segment b"
      style:cursor={editable ? "pointer" : "default"}
      onclick={() => {
        clicked("b");
      }}
      class:active={b}
      style:top={thickness + "px"}
      style:width={thickness + "px"}
      style:height={verticalHeight + "px"}
    >
      b
    </div>
    <div
      class="segment c"
      style:cursor={editable ? "pointer" : "default"}
      onclick={() => {
        clicked("c");
      }}
      class:active={c}
      style:bottom={thickness + "px"}
      style:width={thickness + "px"}
      style:height={verticalHeight + "px"}
    >
      c
    </div>
    <div
      class="segment d"
      style:cursor={editable ? "pointer" : "default"}
      onclick={() => {
        clicked("d");
      }}
      class:active={d}
      style:left={thickness + "px"}
      style:width={horizontalWidth + "px"}
      style:height={thickness + "px"}
    >
      d
    </div>
    <div
      class="segment e"
      style:cursor={editable ? "pointer" : "default"}
      onclick={() => {
        clicked("e");
      }}
      class:active={e}
      style:bottom={thickness + "px"}
      style:width={thickness + "px"}
      style:height={verticalHeight + "px"}
    >
      e
    </div>
    <div
      class="segment f"
      style:cursor={editable ? "pointer" : "default"}
      onclick={() => {
        clicked("f");
      }}
      class:active={f}
      style:top={thickness + "px"}
      style:width={thickness + "px"}
      style:height={verticalHeight + "px"}
    >
      f
    </div>
    <div
      class="segment g"
      style:cursor={editable ? "pointer" : "default"}
      onclick={() => {
        clicked("g");
      }}
      class:active={g}
      style:top={middleTop + "px"}
      style:left={thickness + "px"}
      style:width={horizontalWidth + "px"}
      style:height={thickness + "px"}
    >
      g
    </div>
  </div>
</div>

<style>
  .outer {
    width: calc(100% - 20px);
    height: calc(100% - 20px);
    background-color: rgb(221, 221, 221);
    padding: 10px;
    border-radius: 10px;
  }

  .seven-segment {
    position: relative;
    width: 100%;
    height: 100%;
  }

  .segment {
    display: flex;
    align-items: center;
    justify-content: center;
    position: absolute;
    background-color: #bebebe;
    transition: opacity 0.2s;
    color: rgb(49, 49, 49);
    transition:
      background-color 0.1s ease,
      color 0.1s ease;
    user-select: none;
  }

  .segment.active {
    background-color: #1f1f1f;
    color: rgb(175, 175, 175);
  }

  /* horizontale Segmente */
  .a {
    top: 0;
  }
  .d {
    bottom: 0;
  }

  /* vertikale Segmente */
  .b,
  .c {
    right: 0;
  }
  .e,
  .f {
    left: 0;
  }
</style>
