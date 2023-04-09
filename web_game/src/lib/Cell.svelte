<script>
  export let occupant = "empty";
  export let validPlay = false;

function becomeWhite() {
  occupant = "white";
}
</script>

<button class="cell" on:click="{becomeWhite}" class:validPlay>
<div class="{occupant}" class:validPlay />
</button>

<style>
.cell {
  display: block;
  line-height: 0;

  background: green;
  padding: 0px;
  margin: 0px;
  border-radius: 0;

  width: 50px;
  height: 50px;
}

.white {
  background: white;
  border-radius: 50%;
  width: 100%;
  height: 100%;
}

.black {
  background: black;
  border-radius: 50%;
  width: 100%;
  height: 100%;
}

/* NOTE: if something is validPlay and has a piece on it (never happens), the
   piece will not be shown */

.validPlay {
  background: red;
}
</style>
