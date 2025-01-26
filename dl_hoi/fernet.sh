#!/bin/bash

n_jobs=$(($(nproc) - 1))

helpFunc()
{
    echo ""
    echo "Fernet.sh automatizes the run of backprops configs. It takes a directory with configs (with .json extension) and runs them in parallel."
    echo "Usage: $0 -d directory_with_configs -n number_of_parallel_jobs"
    echo -e "\t -d Directory with configs"
    echo -e "\t -f Python file to run (e.g. expr.py)"
    echo -e "\t -n Number of parallel jobs (default ${n_jobs})"
    echo -e "\t -w Wipe out all .out.txt files in the directory (example: bash fernet.sh -d /path/to/dir -w)"
    echo ""
    exit 1
}

if [ $# -eq 0 ]
then
    helpFunc
fi

while getopts "f:d:n:w?h?" opt
do
    case "$opt" in
        f ) file="$OPTARG" ;;
        d ) dir="$OPTARG" ;;
        n ) n_jobs="$OPTARG" ;;
        w ) rm -f $dir/*.out.txt; exit 0;;
        h | ? | * ) helpFunc ;;
    esac
done

if [ -z "$dir" ]; then
    echo "Error: No se especificó el directoriors."
    helpFunc
fi

if [ -z "$file" ]; then
    echo "Error: No se especificó el archivors."
    helpFunc
fi

if [ ! -d "$dir" ]; then
    echo "Error: Directory $dir does not exist."
    exit 1
fi

echo "About to run ${file} -c ${dir}/*.json"

echo "Running configs (${n_jobs} parallel jobs)"
configs=$(find $dir -name "*.json")

printf "%s\n" "${configs[@]}" | xargs --max-procs=$n_jobs -I % bash -c "python ${file} -c % > %.out.txt 2>&1"
