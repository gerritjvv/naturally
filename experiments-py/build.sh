#!/usr/bin/env bash

CMD="$1"
shift

case "$CMD" in
test)
  poetry run python -m pytest -s -v tests/*
  ;;
run)
  poetry run "$@"
  ;;
*)
  echo "Not supported"
  exit 1
  ;;
esac
