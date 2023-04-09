<script>
  import {setupApi} from './othello.js'
  import Game from './Game.svelte'
</script>

<main>

{#await setupApi()}
Loading....
{:then api}
<Game api={api} />
{/await}

</main>
