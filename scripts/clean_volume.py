import modal

app = modal.App("cleaner")
volume = modal.Volume.from_name("poker-bot-checkpoints", create_if_missing=True)

@app.function(volumes={"/checkpoints": volume})
def clean_volume():
    import os
    import shutil
    
    print("Cleaning volume...")
    for filename in os.listdir("/checkpoints"):
        file_path = os.path.join("/checkpoints", filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            print(f"Deleted {filename}")
        except Exception as e:
            print(f"Failed to delete {filename}. Reason: {e}")
    
    # Commit changes
    volume.commit()
    print("Volume cleaned.")

if __name__ == "__main__":
    with app.run():
        clean_volume.remote()


