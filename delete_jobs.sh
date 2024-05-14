file="delete_jobs.txt"

# delete jobs in txt file
while IFS= read -r job_name; do
    job_name=$(echo "$job_name" | sed 's/[[:space:]]*$//') # Trim spaces at the end
    echo "Deleting job $job_name"
    runai delete job "$job_name"
done < "$file"
