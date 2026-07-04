# AI GENERATED FILE
# SPDX-License-Identifier: Unlicense */

#  This is free and unencumbered software released into the public domain.
# 
#  Anyone is free to copy, modify, publish, use, compile, sell, or
#  distribute this software, either in source code form or as a compiled
#  binary, for any purpose, commercial or non-commercial, and by any
#  means.
# 
#  In jurisdictions that recognize copyright laws, the author or authors
#  of this software dedicate any and all copyright interest in the
#  software to the public domain. We make this dedication for the benefit
#  of the public at large and to the detriment of our heirs and
#  successors. We intend this dedication to be an overt act of
#  relinquishment in perpetuity of all present and future rights to this
#  software under copyright law.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
#  OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#  ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
#  OTHER DEALINGS IN THE SOFTWARE.
# 
#  For more information, please refer to <http://unlicense.org/>

#/bin/bash

# Base directory where all dataset folders are located
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting dataset downloads and extraction..."

for D in "$BASE_DIR"/*/; do
    if [ -f "${D}url.txt" ]; then
        URL=$(cat "${D}url.txt")
        FILENAME=$(basename "$URL")
        DIR_NAME=$(basename "$D")
        
        echo "Processing $DIR_NAME..."
        
        (
            cd "$D" || exit
            
            # 1. Download
            if [ ! -f "$FILENAME" ]; then
                echo "Downloading $FILENAME to $DIR_NAME..."
                wget -q --show-progress "$URL" -O "$FILENAME"
            else
                echo "File $FILENAME already exists in $DIR_NAME, skipping download."
            fi
            
            # 2. Extract
            echo "Extracting $FILENAME in $DIR_NAME..."
            mkdir -p .tmp_extract
            
            if [[ "$FILENAME" == *.tar.gz ]]; then
                tar -xzf "$FILENAME" -C .tmp_extract
            elif [[ "$FILENAME" == *.zip ]]; then
                unzip -q -o "$FILENAME" -d .tmp_extract
            else
                echo "Unknown file extension for $FILENAME. Skipping extraction."
                rm -rf .tmp_extract
                exit 0
            fi
            
            # 3. Move contents up if they were extracted into a single subdirectory
            shopt -s dotglob
            CONTENTS=(.tmp_extract/*)
            if [ ${#CONTENTS[@]} -eq 1 ] && [ -d "${CONTENTS[0]}" ]; then
                # Move contents of that single directory to the current folder
                mv "${CONTENTS[0]}"/* . 2>/dev/null
            else
                # Move everything from .tmp_extract to the current folder
                mv .tmp_extract/* . 2>/dev/null
            fi
            shopt -u dotglob
            
            rm -rf .tmp_extract
            
            echo "Finished processing $DIR_NAME."
        ) &
    fi
done

echo "All background jobs started. Waiting for completion..."
wait
echo "All datasets have been successfully processed."
