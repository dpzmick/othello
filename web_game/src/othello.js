import Module from "../../emcc-build/web_game/wasm_wrapper.js"

export async function setupApi() {
  const othelloMod = await Module();
  return {
    new: othelloMod.cwrap("new_othello_wrap", "number", ["number"]),

    turn: othelloMod.cwrap("othello_wrap_turn", "number", ["number"]),

    board_at: othelloMod.cwrap("othello_wrap_board_at",
                               "number", ["number", "number", "number"]),

    play_at: othelloMod.cwrap("othello_wrap_play_at",
                              "number", ["number", "number", "number"]),
  };
}

export function OthelloGame(api, aiType) {
  let g = api.new(aiType);

  return {
    turn: function(x,y) {
      let t = api.turn(g);
      if (t==0) return "white";
      if (t==1) return "black";
    },

    boardAt: function(x,y) {
      let t = api.board_at(g, x, y);
      if (t == 1) return "white";
      if (t == -1) return "black";
      if (t == 0) return "empty";
      if (t == 2) return "valid";
    },

    getScores: function() {
      let whiteScore = 0;
      let blackScore = 0;

      // just tracking/computing this ourselves for ease
      for (let y = 0; y < 8; ++y) {
        for (let x = 0; x < 8; ++x) {
          let cell = this.boardAt(x,y);
          whiteScore += cell=="white";
          blackScore += cell=="black";
        }
      }

      return [whiteScore, blackScore];
    },

    // return nicely formatted cells array for the ui
    getCells: function(x,y) {
      let cells = [];
      for (let y = 0; y < 8; ++y) {
        let row = [];
        for (let x = 0; x < 8; ++x) {
          let cell = this.boardAt(x,y);
          row.push(cell);
        }
        cells.push(row);
      }
      return cells;
    },

    playAt: function(x,y) {
      api.play_at(g, x, y);
    },
  };
}
