{
  description = "CVM and MANTIS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane = {
      url = "github:ipetkov/crane";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    cosmos = {
      url = "github:informalsystems/cosmos.nix";
      inputs.nixpkgs.follows = "nixpkgs";
      inputs.rust-overlay.follows = "rust-overlay";
    };
    datamodel-code-generator-src = {
      url = "github:koxudaxi/datamodel-code-generator";
      flake = false;
    };
    poetry2nix = {
      url = "github:nix-community/poetry2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    networks = {
      url = "github:ComposableFi/networks";
    };
    cosmpy-src = {
      url = "github:fetchai/cosmpy";
      flake = false;
    };

    fastapi-cache-src = {
      url = "github:long2ice/fastapi-cache";
      flake = false;
    };

    devour-flake = {
      url = "github:srid/devour-flake";
      flake = false;
    };
  };

  outputs =
    inputs @ { flake-parts
    , self
    , crane
    , devour-flake
    , datamodel-code-generator-src
    , poetry2nix
    , cosmpy-src
    , ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [
      ];

      systems = [ "x86_64-linux" "aarch64-linux" "aarch64-darwin" "x86_64-darwin" ];
      perSystem =
        { config
        , self'
        , inputs'
        , pkgs
        , system
        , ...
        }:
        let
          craneLib = crane.lib.${system};
          rust-src = pkgs.lib.cleanSourceWith {
            filter =
              pkgs.nix-gitignore.gitignoreFilterPure
                (
                  name: type:
                    !(pkgs.lib.strings.hasSuffix ".nix" name)
                    || builtins.match ".*proto$" name != null
                    || builtins.match ".*txt$" name != null
                    || craneLib.filterCargoSources name type
                ) [ ./.gitignore ]
                ./.;
            src = craneLib.path ./.;
          };
          devour-flake = pkgs.callPackage inputs.devour-flake { };
          rust-toolchain =
            pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
          rust =
            (self.inputs.crane.mkLib pkgs).overrideToolchain
              rust-toolchain;
          makeCosmwasmContract = name: rust: std-config:
            let
              binaryName = "${builtins.replaceStrings ["-"] ["_"] name}.wasm";
              maxWasmSizeBytes = 819200;
              profile = "deployment";
            in
            rust.buildPackage (rust-attrs
              // {
              src = rust-src;
              pnameSuffix = "-${name}";
              nativeBuildInputs = [
                pkgs.binaryen
                self.inputs.cosmos.packages.${system}.cosmwasm-check
              ];
              pname = name;
              cargoBuildCommand = "cargo build --target wasm32-unknown-unknown --profile ${profile} --package ${name} ${std-config}";
              RUSTFLAGS = "-C link-arg=-s";
              installPhaseCommand = ''
                mkdir --parents $out/lib
                # from CosmWasm/rust-optimizer
                # --signext-lowering is needed to support blockchains runnning CosmWasm < 1.3. It can be removed eventually
                wasm-opt target/wasm32-unknown-unknown/${profile}/${binaryName} -o $out/lib/${binaryName} -Os --signext-lowering
                cosmwasm-check $out/lib/${binaryName}
                SIZE=$(stat --format=%s "$out/lib/${binaryName}")
                if [[ "$SIZE" -gt ${builtins.toString maxWasmSizeBytes} ]]; then
                  echo "Wasm file size is $SIZE, which is larger than the maximum allowed size of ${builtins.toString maxWasmSizeBytes} bytes."
                  echo "Either reduce size or increase maxWasmSizeBytes if you know what you are doing."
                  exit 1
                fi
              '';
            });

          rust-attrs = {
            doCheck = false;
            checkPhase = "true";
            cargoCheckCommand = "true";
            NIX_BUILD_FLAKE = "true";
            RUST_BACKTRACE = "full";
            CARGO_PROFILE_RELEASE_BUILD_OVERRIDE_DEBUG = true;
            buildInputs = [ pkgs.protobuf ];
          };
          cw-cvm-outpost = makeCosmwasmContract "cw-cvm-outpost" rust "--no-default-features --features=std,json-schema,cosmos";
          cw-cvm-executor = makeCosmwasmContract "cw-cvm-executor" rust "--no-default-features --features=std,json-schema,cosmos";
          cw-mantis-order = makeCosmwasmContract "cw-mantis-order" rust "--no-default-features --features=std,json-schema";
          cosmwasm-contracts = pkgs.symlinkJoin {
            name = "cosmwasm-contracts";
            paths = [
              cw-cvm-executor
              cw-cvm-outpost
              cw-mantis-order
            ];
          };
          cosmwasm-json-schema-ts = pkgs.writeShellApplication {
            name = "cosmwasm-json-schema-ts";
            runtimeInputs = with pkgs; [
              rust
              nodejs
              nodePackages.npm
            ];
            text = ''
              echo "generating TypeScript types and client definitions from JSON schema of CosmWasm contracts"
              cd contracts/cosmwasm/cvm-runtime
              npm install
              rm --recursive --force dist

              rm --recursive --force schema
              cargo run --bin order --package cw-mantis-order
              npm run build-cw-mantis-order

              rm --recursive --force schema
              cargo run --bin outpost --package xc-core
              npm run build-xc-core

              npm publish
            '';
          };
        in
        let
          datamodel-code-generator = mkPoetryApplication {
            projectDir = datamodel-code-generator-src;
            checkGroups = [ ];
          };

          cosmpy = pkgs.python3Packages.buildPythonPackage {
            name = "cosmpy";
            version = "0.9.1";
            format = "pyproject";

            src = cosmpy-src;

            nativeBuildInputs = [
              pkgs.python3Packages.poetry-core
            ];
          };

          fastapi-cache2 = pkgs.python3Packages.buildPythonPackage {
            name = "fastapi-cache2";
            version = "0.2.1";
            format = "pyproject";

            src = inputs.fastapi-cache-src;

            nativeBuildInputs = [
              pkgs.python3Packages.poetry-core
            ];
          };

          python-packages = ps: with ps; [ numpy cvxpy wheel virtualenv uvicorn fastapi pydantic environs fastapi-cache2 pip cosmpy pendulum jsonschema grpcio ecdsa bech32 requests protobuf python-dateutil pycryptodome googleapis-common-protos ];
          python = pkgs.python3.withPackages python-packages;
          inherit (poetry2nix.lib.mkPoetry2Nix { inherit pkgs; }) mkPoetryApplication mkPoetryPackages;
          env = {
            OSMOSIS_POOLS = "https://app.osmosis.zone/api/pools?page=1&limit=1000&min_liquidity=500000";
            ASTROPORT_POOLS = "https://app.astroport.fi/api/trpc/pools.getAll?input=%7B%22json%22%3A%7B%22chainId%22%3A%5B%22neutron-1%22%5D%7D%7D";
            SKIP_MONEY_SWAGGER = "https://api-swagger.skip.money/";
            SKIP_MONEY = "https://api.skip.money/";
          };
          cosmwasm-json-schema-py =
            let
            in
            pkgs.writeShellApplication {
              name = "cosmwasm-json-schema-py";
              runtimeInputs = with pkgs; [
                rust.cargo
                datamodel-code-generator
              ];
              text = ''
                RUST_BACKTRACE=1 cargo run --package cvm-runtime --bin outpost --features=cosmwasm,json-schema,cosmos,std

                datamodel-codegen  --input schema/raw/ --input-file-type jsonschema --output mantis/blackbox/cvm_runtime/  --disable-timestamp --target-python-version "3.10" --use-schema-description --output-model-type "pydantic.BaseModel"

                curl "${env.OSMOSIS_POOLS}" | jq .pools > schema/osmosis_pools.json
                datamodel-codegen  --input schema/osmosis_pools.json --input-file-type json --output mantis/blackbox/osmosis_pools.py  --disable-timestamp --target-python-version "3.10" --use-schema-description --output-model-type "pydantic.BaseModel"

                curl "${env.ASTROPORT_POOLS}" | jq .result.data > schema/neutron_pools.json
                datamodel-codegen  --input schema/neutron_pools.json --input-file-type json --output mantis/blackbox/neutron_pools.py  --disable-timestamp --target-python-version "3.10" --use-schema-description --output-model-type "pydantic.BaseModel"

                curl "${env.SKIP_MONEY_SWAGGER}swagger.yml" > schema/skip_money_swagger.yml
                datamodel-codegen  --input schema/skip_money_swagger.yml --input-file-type openapi --output mantis/blackbox/skip_money.py  --disable-timestamp --target-python-version "3.10" --use-schema-description --output-model-type "pydantic.BaseModel"
              '';
            };
        in
        {
          _module.args.pkgs = import self.inputs.nixpkgs {
            inherit system;
            overlays = with self.inputs; [
              rust-overlay.overlays.default
            ];
          };
          devShells.default = pkgs.mkShell {
            VIRTUALENV_PYTHON = "${python}/bin/python3.11";
            VIRTUAL_ENV = 1;
            nativeBuildInputs = [ python pkgs.cbc ];
            buildInputs = [
              python
              devour-flake
              pkgs.virtualenv
              pkgs.conda
              pkgs.pyo3-pack
              rust.cargo
              rust.rustc
              devour-flake
            ];
            shellHook = ''
              if [[ -f ./.env ]]; then
                source ./.env
              fi
            '';
          };
          formatter = pkgs.alejandra;
          packages = rec {
            inherit cw-mantis-order cw-cvm-executor cw-cvm-outpost cosmwasm-contracts cosmwasm-json-schema-py datamodel-code-generator;

            mantis = rust.buildPackage (rust-attrs
              // {
              src = rust-src;
              pname = "mantis";
              name = "mantis";
              cargoBuildCommand = "cargo build --release --bin mantis";
              nativeBuildInputs = [ pkgs.cbc ];
            });
            default = mantis-blackbox;
            mantis-blackbox = pkgs.writeShellApplication {
              name = "run";
              runtimeInputs = [
                python
                pkgs.cbc
              ];
              text = ''
                # cd $ {./mantis}
                cd mantis
                OSMOSIS_POOLS="${env.OSMOSIS_POOLS}"
                ASTROPORT_POOLS="${env.ASTROPORT_POOLS}"
                SKIP_MONEY="${env.SKIP_MONEY}"
                COMPOSABLE_COSMOS_GRPC="${inputs.networks.lib.pica.mainnet.GRPC}"
                CVM_ADDRESS="${inputs.networks.lib.pica.mainnet.CVM_OUTPOST_CONTRACT_ADDRESS}"

                export SKIP_MONEY
                export OSMOSIS_POOLS
                export ASTROPORT_POOLS
                export COMPOSABLE_COSMOS_GRPC
                export CVM_ADDRESS
                uvicorn blackbox.main:app --reload --log-level trace --host "0.0.0.0"
              '';
            };
            ci = pkgs.writeShellApplication {
              name = "nix-build-all";
              runtimeInputs = [
                pkgs.nix
                devour-flake
              ];
              text = ''
                nix flake lock --no-update-lock-file
                devour-flake . "$@"
              '';
            };
          };
        };
    };
}
