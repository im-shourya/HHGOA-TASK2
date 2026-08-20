#!/usr/bin/env zsh
files=($(git ls-files --others --exclude-standard))
total=${#files[@]}
chunks=12
size=$(( (total + chunks - 1) / chunks ))

for ((i=0; i<chunks; i++)); do
    start=$((i * size + 1))
    if [[ $start -gt $total ]]; then break; fi
    end=$((start + size - 1))
    if [[ $end -gt $total ]]; then end=$total; fi
    
    # Note zsh arrays are 1-indexed
    chunk=("${(@)files[start,end]}")
    if [[ ${#chunk[@]} -gt 0 ]]; then
        git add "${chunk[@]}"
        git commit -m "Commit $((i+1)) of $chunks"
    fi
done

git push origin HEAD
