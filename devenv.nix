{ pkgs, lib, config, inputs, ... }:

{

  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages = [
    pkgs.git
    pkgs.python312Packages.torch
    pkgs.python312Packages.torchvision 
  ];

  # https://devenv.sh/languages/
  languages.python.enable = true;
  languages.python.version = "3.12.8";
  languages.python.venv.enable = true;
  languages.python.venv.requirements = ./requirements.txt;

  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  enterShell = ''
    zsh
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';
}