from huggingface_hub import snapshot_download

print("Starting download for facebook/bart-base...")
print("This may take a few minutes. It will handle any network drops automatically.")

# This forces the download to your local system cache
snapshot_download(repo_id="facebook/bart-base")

print("\nDownload complete! The model is safely cached.")