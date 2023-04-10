<script>
  import {OthelloGame} from './othello.js'

  export let api;

  // gloal params
  let aiType = 0;
  let game = OthelloGame(api, aiType);
  let cells = game.getCells()
  let [whiteScore, blackScore] = game.getScores();
  let player = game.turn();

  let dialog; // Reference to the dialog tag
  const closeClick = () => {
    dialog.close();
  };

  const restartGame = () => {
    let e = document.getElementById("ai-type");
    let selected = e.value;

    aiType = selected;
    game = OthelloGame(api, aiType);
    cells = game.getCells()
    let [_whiteScore, _blackScore] = game.getScores(); // binding directly not compiling right?
    whiteScore = _whiteScore;
    blackScore = _blackScore;
    player = game.turn();

    dialog.close();
  };

  function makePlay(x,y) {
    game.playAt(x,y);
    cells = game.getCells();
    [whiteScore, blackScore] = game.getScores(); // for some reason this works here
    player = game.turn();
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
    <option value="2" selected={aiType=="2"}>Monte Carlo Tree Search w/ Neural Net</option>
    <option value="3" selected={aiType=="3"}>No AI</option>
  </select>
  <button on:click={restartGame}>Restart Game</button>
  <button on:click={closeClick}>Close Settings</button>
</dialog>

<p>White: {whiteScore}. Black: {blackScore}</p>
<p>{player} to play</p>
<table>
  {#each cells as rowCells, y}
    <tr>
      {#each rowCells as cell, x}
        <td>
          <button class="cell" on:click={() => makePlay(x, y)}>
            <div class={cell} />
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
