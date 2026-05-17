<script>
  import {OthelloGame} from './othello.js'

  export let api;

  let aiType = "0";
  let game = OthelloGame(api, aiType);
  let cells = game.getCells()
  let [whiteScore, blackScore] = game.getScores();
  let player = game.turn();
  let nValid = game.nValidMoves();
  let isOver = game.gameOver();

  function refreshState() {
    cells = game.getCells();
    [whiteScore, blackScore] = game.getScores();
    player = game.turn();
    nValid = game.nValidMoves();
    isOver = game.gameOver();
  }

  let dialog; // Reference to the dialog tag
  const closeClick = () => {
    dialog.close();
  };

  const restartGame = () => {
    let e = document.getElementById("ai-type");
    let selected = e.value;

    aiType = selected;
    game = OthelloGame(api, aiType);
    refreshState();

    dialog.close();
  };

  function makePlay(x,y) {
    game.playAt(x,y);
    refreshState();
  }

  // When the human has no legal moves but the game isn't over, the only
  // available action is "pass". Any cell coordinate works -- the C wrapper
  // auto-passes whenever own_moves is empty.
  function passTurn() {
    game.playAt(0, 0);
    refreshState();
  }
</script>

<div id="game">
<h3>Othello</h3>
<dialog id="configuration-dialog" bind:this={dialog}>
  <h1>Game Setup</h1>
  <p>Select type of AI to use:</p>
  <select name="ai-type" id="ai-type">
    <option value="0" selected={aiType=="0"}>Monte Carlo Tree Search</option>
    <option value="1" selected={aiType=="1"}>Neural Net</option>
    <option value="3" selected={aiType=="3"}>No AI</option>
  </select>
  <button on:click={restartGame}>Restart Game</button>
  <button on:click={closeClick}>Close Settings</button>
</dialog>

<p>White: {whiteScore}. Black: {blackScore}</p>
{#if isOver}
  {#if whiteScore > blackScore}
    <p><strong>Game over — white wins.</strong></p>
  {:else if blackScore > whiteScore}
    <p><strong>Game over — black wins.</strong></p>
  {:else}
    <p><strong>Game over — tied.</strong></p>
  {/if}
{:else}
  <p>{player} to play{#if nValid === 0} (no legal moves — must pass){/if}</p>
  {#if nValid === 0}
    <button on:click={passTurn}>Pass</button>
  {/if}
{/if}
<table>
  {#each cells as rowCells, y}
    <tr>
      {#each rowCells as cell, x}
        <td>
          <button class="cell" on:click={() => makePlay(x, y)}>
            <div class={cell}></div>
          </button>
        </td>
      {/each}
    </tr>
  {/each}
</table>

<p></p>
<button on:click={() => dialog.showModal()}>Configure Game</button>
</div>

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
