#!/bin/bash

# This is a helper script to generate a JSON file listing all versions of zonopy's documentation
# to be used by github actions during a release to update the version dropdown in the documentation.

# Directory to scan for versions
DIRECTORY="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Base URL for your site
BASE_URL="https://roahmlab.github.io/zonopy/"

# Initialize the JSON array with special versions manually if they exist
json_array="["

# Add special versions if they exist
if [ -d "$DIRECTORY/dev" ]; then
  json_array+="{\"name\": \"devel\", \"version\": \"dev\", \"url\": \"${BASE_URL}dev/\"},"
fi

# Collect all version directories excluding special ones
versions=()
for dir in $(find $DIRECTORY -maxdepth 1 -mindepth 1 -type d ! -name 'dev' ! -name '.git' -exec basename {} \; | sort -Vr); do
  versions+=("$dir")
done

# Determine the latest version for naming
latest_version=${versions[0]}

# Loop through the versions array to build JSON objects
for version in "${versions[@]}"; do
    if [ "$version" == "$latest_version" ]; then
        # Mark the latest version with a special name
        json_array+="{\"name\": \"latest ($version)\", \"version\": \"$version\", \"url\": \"${BASE_URL}${version}/\", \"preferred\": true},"
    else
        json_array+="{\"version\": \"$version\", \"url\": \"${BASE_URL}${version}/\"},"
    fi
done

# Remove the last comma to properly close the JSON array
json_array=${json_array%,}

# Close the JSON array
json_array+="]"

# Output the JSON array
echo $json_array | python3 -m json.tool > "$DIRECTORY/versions.json"
