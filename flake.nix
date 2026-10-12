{
  description = "melissa package";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    sbi = {
      url = "github:Filippo-Galli/sbi";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      sbi,
      ...
    }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = nixpkgs.lib.genAttrs systems;

    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
          };

        in
        {
          markovsbi = pkgs.python3Packages.callPackage ./. {
          };

          default = self.packages.${system}.markovsbi;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
          };
        in
        {
          default = pkgs.callPackage ./shell.nix {
            markovsbi = self.packages.${system}.markovsbi;
            sbi = sbi.packages.${system}.sbi;
          };
        }
      );
    };
}
