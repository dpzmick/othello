<script>
  let whiteScore = 0;
  let blackScore = 0;

  function getCells() {
    whiteScore=0;
    blackScore=0;

    let cells = [];
    for (let y = 0; y < 8; ++y) {
      let row = [];
      for (let x = 0; x < 8; ++x) {
        let cell = game.boardAt(x,y);
        whiteScore += cell=="white";
        blackScore += cell=="black";
        row.push(cell);
      }
      cells.push(row);
    }
    return cells;
  }

  function makePlay(x, y) {
    game.playAt(x,y);
    // oponent picks move
    cells = getCells();
  }

  export let game = null; // FIXME make required
  let cells = getCells();


</script>

<p>White: {whiteScore}. Black: {blackScore}</p>
<table>
  {#each cells as rowCells, y}
    <tr>
      {#each rowCells as cell, x}
        <td>
          <button class="cell"
            on:click="{() => makePlay(x, y)}">
              <div class={cell} />
          </button>
        </td>
      {/each}
    </tr>
  {/each}
</table>

<style>
table {
  border-collapse: collapse;
}

td {
  padding: 0;
  border: 1px solid black;
}

.cell {
  display: block;
  line-height: 0;

  background: green;
  padding: 0px;
  border-radius: 0;

  width: 50px;
  height: 50px;
}

.white {
  display: block;
  margin: auto;
  background: white;
  border-radius: 50%;
  width: 100%;
  height: 100%;
}

.black {
  display: block;
  margin: auto;
  background: black;
  border-radius: 50%;
  width: 100%;
  height: 100%;
}

.valid {
  display: block;
  margin: auto;
  background: red;
  border-radius: 50%;
  width: 30%;
  height: 30%;
}
</style>
