import Module from "../../emcc-build/web_game/wasm_wrapper.js"

async function setupApi() {
  const othelloMod = await Module();
  return {
    new: othelloMod.cwrap("new_othello_wrap", "number", ["number"]),

    board_at: othelloMod.cwrap("othello_wrap_board_at",
                               "number", ["number", "number", "number"]),

    play_at: othelloMod.cwrap("othello_wrap_play_at",
                              "number", ["number", "number", "number"]),
  };
}

export async function OthelloGame() {
  let api = await setupApi();
  let g = api.new(1);

  return {
    boardAt: function(x,y) {
      let t = api.board_at(g, x, y);
      if (t == 1) return "white";
      if (t == -1) return "black";
      if (t == 0) return "empty";
      if (t == 2) return "valid";
    },

    playAt: function(x,y) {
      api.play_at(g, x, y);
    },
  };
}
