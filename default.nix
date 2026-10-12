{
  lib,
  buildPythonPackage,
  setuptools,
  wheel,
  numpy,
  matplotlib,
  jax,
  optax,
  dm-haiku,
  jupyter,
  pillow,
  seaborn,
  jaxtyping,
  blackjax,
  chex,
}:
buildPythonPackage (finalAttrs: {
  pname = "markovsbi";
  version = "0-unstable-2025-02-28";
  __structuredAttrs = true;

  src = lib.fileset.toSource {
    root = ./.;
    fileset = lib.fileset.unions [
      ./markovsbi
      ./setup.py
    ];
  };

  pyproject = true;
  dontUseCmakeConfigure = true;

  build-system = [
    setuptools
    wheel
  ];

  dependencies = [
    numpy
    matplotlib
    jax
    optax
    dm-haiku
    jupyter
    pillow
    seaborn
    jaxtyping
    blackjax
    chex
  ];

  pythonImportsCheck = [
    "markovsbi"
    "markovsbi.utils"
    "markovsbi.tasks"
  ];
  doCheck = false;

})
