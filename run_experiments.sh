#!/bin/sh

# Path to the input file
input_file="/nfs/home/mboels/projects/SuPRA/run_best_methods.txt"

# Check if the file exists
if [ ! -f "$input_file" ]; then
    echo "Error: Input file $input_file not found."
    exit 1
fi

# Create a temporary file to store experiment names
temp_file=$(mktemp)

# Extract experiment names to the temporary file
while IFS= read -r line; do
    # Skip empty lines or lines starting with '\\'
    if [ -z "$line" ] || echo "$line" | grep -q "^\\\\"; then
        continue
    fi
    
    # If we get here, the line is not a comment and not empty - it should be an experiment name
    echo "$line" >> "$temp_file"
done < "$input_file"

# Show all experiments that will be run
echo "The following experiments will be run:"
while IFS= read -r exp; do
    echo "  - $exp"
done < "$temp_file"

# Ask for confirmation
printf "Do you want to proceed? (y/n): "
read confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Operation cancelled."
    rm "$temp_file"
    exit 0
fi

# Run all experiments
echo "Starting to run experiments..."
while IFS= read -r exp; do
    echo "-----------------------------------------------------------"
    echo "Sending: sh runai.sh $exp to the cluster..."
    sh runai.sh "$exp"
    sleep 2
done < "$temp_file"

# Clean up
rm "$temp_file"

echo "All experiments have been run."
