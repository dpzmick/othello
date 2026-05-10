{
  description = "Othello engine, MCTS/NN players, and ML training pipeline";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        isDarwin = pkgs.stdenv.isDarwin;
        isLinux = pkgs.stdenv.isLinux;

        nativeBuildDeps = with pkgs; [
          cmake
          gnumake
          pkg-config
          zstd
        ];

        webBuildDeps = with pkgs; [
          emscripten
          nodejs_20
        ];

        pythonDeps = with pkgs; [
          uv
          python311
        ];

        devTools = with pkgs; [
          git
          ripgrep
          fd
        ] ++ lib.optionals isLinux [
          gdb
          valgrind
        ] ++ lib.optionals isDarwin [
          lldb
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = nativeBuildDeps ++ webBuildDeps ++ pythonDeps ++ devTools;

          shellHook = ''
            # Emscripten in nixpkgs ships a fully-resolved .emscripten config
            # but its cache directory inside the store is read-only. Copy the
            # config locally and point CACHE at a writable directory.
            export EM_CACHE="$PWD/.emscripten-cache"
            export EM_CONFIG="$PWD/.emscripten-config"
            mkdir -p "$EM_CACHE"
            if [ ! -f "$EM_CONFIG" ]; then
              cp ${pkgs.emscripten}/share/emscripten/.emscripten "$EM_CONFIG"
              chmod +w "$EM_CONFIG"
              echo "CACHE = '$EM_CACHE'" >> "$EM_CONFIG"
            fi

            # uv: keep the venv inside the project so direnv etc. pick it up.
            export UV_PROJECT_ENVIRONMENT="$PWD/.venv"

            echo
            echo "othello dev shell ready"
            echo "  native build:   cmake -B build -S . && cmake --build build -j"
            echo "  run tests:      ./build/tests/test"
            echo "  web (dev):      cd web_game && npm install && npm run dev"
            echo "  web (wasm):     emcmake cmake -B emcc-build -S . && cmake --build emcc-build -j"
            echo "  python setup:   uv sync"
            echo "  python run:     uv run python -m othello_ml.pipeline"
            echo
          '';
        };

        formatter = pkgs.nixpkgs-fmt;
      });
}
