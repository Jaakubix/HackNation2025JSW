#!/usr/bin/env bash
set -eo pipefail

# Instalacja środowiska conda z pliku environment.yml (fallback: requirements.txt)
# Użycie: ./install_env.sh [nazwa_env] [python_version]
# Domyślnie: nazwa_env=kopalnia, python_version=3.12

ENV_NAME="${1:-kopalnia}"
PY_VER="${2:-3.12}"
CONDA_ROOT="${CONDA_ROOT:-/home/jakub-pytka/miniconda3}"
ENV_FILE="${ENV_FILE:-plan/environment.yml}"

if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
  echo "Nie znaleziono conda w ${CONDA_ROOT}/bin/conda. Zainstaluj Minicondę i uruchom ponownie." >&2
  exit 1
fi

# Załaduj conda do bieżącej powłoki
eval "$("${CONDA_ROOT}/bin/conda" shell.bash hook)"

# Jeśli mamy plik environment.yml, użyj go do stworzenia/aktualizacji env
if [[ -f "${ENV_FILE}" ]]; then
  if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
  else
    conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"
  fi
else
  # Fallback: ręczne stworzenie env i instalacja z requirements.txt
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PY_VER}"
  fi
  conda activate "${ENV_NAME}"
  pip install --upgrade pip
  pip install -r requirements.txt
  echo "Środowisko '${ENV_NAME}' gotowe. Aktywuj: conda activate ${ENV_NAME}"
  exit 0
fi

conda activate "${ENV_NAME}"
echo "Środowisko '${ENV_NAME}' gotowe. Aktywuj: conda activate ${ENV_NAME}"
