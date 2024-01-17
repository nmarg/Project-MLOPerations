from google.cloud import storage


def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)


upload_to_gcs("project-mloperations-data", "data/drifting/current_data.csv", "data/drifting/current_data.csv")
