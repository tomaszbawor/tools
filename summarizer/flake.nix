{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let pkgs = import nixpkgs { inherit system; };
      in
      rec {

        # ───── package that `nix build` produces ─────
        packages.default = pkgs.buildNpmPackage {
          pname = "ollama-meeting-summariser";
          version = "0.1.0";
          src = self; # project root

          # auto-fill after first build failure (`nix build` tells you the hash).
          npmDepsHash = "sha256-VrkHlCufMgFA/Wk/ycJeMFl1EmH2qtLCpZA/XKr22PU=";

          buildPhase = ''
            runHook preBuild
            # compile TS → JS
            npx tsc --project tsconfig.json
            runHook postBuild
          '';


          installPhase = ''
            mkdir -p $out/bin
            # 1️⃣  copy compiled script
            cp build/summarise.js $out/bin/
            # 2️⃣  copy ALL runtime deps next to the script
            cp -R node_modules $out/bin/
            # 3️⃣  create wrapper: add Ollama to PATH *and*
            #     teach Node where to look for those deps
            makeWrapper ${pkgs.nodejs_latest}/bin/node $out/bin/summarise \
              --add-flags $out/bin/summarise.js \
              --set NODE_PATH $out/bin/node_modules \
              --prefix PATH : ${pkgs.ollama}/bin
          '';

          meta.mainProgram = "summarise";
        };

        # ───── dev shell for hacking ─────
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [ nodejs_latest ollama gcc ];
        };
      });
}
