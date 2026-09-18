# Audit scripts

Read-only audit programs will be added after the image archive is received and its structure is inspected. Each script must accept paths as arguments, write only to `audit_results/`, and never modify the dataset or checkpoint.
