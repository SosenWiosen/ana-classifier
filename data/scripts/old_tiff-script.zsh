#!/bin/zsh

src="/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D"
dst="/Users/sosen/UniProjects/eng-thesis/data/data-uncompressed/2D-tiff"

find "$src" -name "*.tiff" -type f -print | while IFS= read -r file; do
  filepath="${file#$src}"
  dirpath="$dst/${filepath%/*}"
  mkdir -p "$dirpath"
  cp "$file" "$dirpath"
done
