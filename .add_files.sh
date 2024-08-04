#!/bin/bash

MAX_SIZE=100000000  # Max file size in bytes (adjust as needed)

for file in $(find . -type f); do
  if [ $(stat -c%s "$file") -le $MAX_SIZE ]; then
    git add "$file"
  else
    echo "Skipping large file: $file"
  fi
done
