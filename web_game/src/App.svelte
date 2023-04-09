<script>
  import Game from './Game.svelte'
  import {OthelloGame} from './othello.js'
</script>

<main>
{#await OthelloGame()}
Loading....
{:then game}
  <h1>Othello</h1>
  <Game game={game} />
{/await}
</main>
