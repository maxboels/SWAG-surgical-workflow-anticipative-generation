file="delete_jobs.txt"

# Trim every line in the file at the first space
sed -i 's/ .*//' "$file"

# delete jobs in txt file
while IFS= read -r job_name; do
    # Remove carriage return
    job_name=$(echo "$job_name" | tr -d '\r')
    echo "Deleting job $job_name"
    runai delete job "$job_name"
done < "$file"
