from google.cloud import bigquery

PROJECT_ID = "bigquery-finance-ia"
path_to_json = "/Users/jackdo/Downloads/bigquery-finance-ia-1972976e09a1.json"

# On initialise le client en lui fournissant directement la clé JSON
client = bigquery.Client.from_service_account_json(path_to_json, project=PROJECT_ID)