{
  mkShell,
  ruff,
  ty,
  uv,
  marimo,
  nixfmt,
  python3Packages,
  sbi,
  markovsbi,
  nodejs,
}:
mkShell {
  packages = [
    ruff
    ty
    uv
    nixfmt
    python3Packages.numpy
    python3Packages.matplotlib
    python3Packages.jax
    python3Packages.optax
    python3Packages.dm-haiku
    python3Packages.jupyter
    python3Packages.pillow
    python3Packages.seaborn
    python3Packages.jaxtyping
    python3Packages.torch
    python3Packages.blackjax
    marimo
    nodejs
    markovsbi
    sbi
  ];

  shellHook = ''
    export PYTHONPATH="$PWD:$PYTHONPATH"
  '';
}
