#!/bin/bash

shopt -s nocaseglob

echo "🚀 Running with updated Tilt mappings..."

# 1. Clean up placeholders
find . -type f -iname "placeholder.txt" -delete

# 2. Process Folders
find . -type d | while read -r dir; do
    [ "$dir" == "." ] && continue

    declare -A count
    # Initializing all possible categories to 1
    for cat in Uncovered Covered Very_Covered Low_Tilt Tilted Slight_Tilt Very_Tilted; do count[$cat]=1; done

    for file in "$dir"/*; do
        [ -f "$file" ] || continue
        [[ "$file" == *"$0"* ]] && continue

        filename=$(basename "$file")
        upper_name=$(echo "$filename" | tr '[:lower:]' '[:upper:]')
        category=""

        # --- Updated Logic Gates ---
        if [[ $upper_name == NC* ]] || [[ $upper_name == UNCOVERED* ]]; then
            category="Uncovered"
        elif [[ $upper_name == MC* ]] || [[ $upper_name == VC* ]] || [[ $upper_name == VERY_COVERED* ]]; then
            category="Very_Covered"
        elif [[ $upper_name == SC* ]] || [[ $upper_name == COVERED* ]]; then
            category="Covered"

        # --- Adjusted Tilt Logic ---
        elif [[ $upper_name == TT* ]] || [[ $upper_name == NT* ]] || [[ $upper_name == LOW_TILT* ]]; then
            category="Low_Tilt"
        elif [[ $upper_name == MT* ]] || [[ $upper_name == TILTED* ]]; then
            category="Tilted"
        elif [[ $upper_name == ST* ]] || [[ $upper_name == SLIGHT_TILT* ]]; then
            category="Slight_Tilt"
        elif [[ $upper_name == VT* ]] || [[ $upper_name == VERY_TILT* ]]; then
            category="Very_Tilted"
        fi

        if [ -n "$category" ]; then
            new_name="${category}_${count[$category]}.jpg"
            target_path="$dir/$new_name"

            # Conversion logic
            if magick "$file" -quality 95 "$target_path"; then
                # Delete original only if the paths are different
                if [ "$(realpath "$file")" != "$(realpath "$target_path")" ]; then
                    rm "$file"
                fi
                ((count[$category]++))
            else
                echo "❌ Fedora cannot decode $filename - Try: sudo dnf install libheif-freeworld ImageMagick-heic"
            fi
        fi
    done
done

# 3. Apply Blurring (Excluding Identity 9)
echo "🌫️ Applying Blur..."
find . -path "./Identity 9/Blur" -prune -o -type f -name "*Slight_Blur_*" -exec magick "{}" -blur 0x1 "{}" \;
find . -path "./Identity 9/Blur" -prune -o -type f \( -name "*Very_Blurry_*" -o -name "*Very_Blury_*" \) -exec magick "{}" -blur 0x5 "{}" \;

echo "✅ Mappings updated and files processed!"
