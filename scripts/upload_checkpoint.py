import modal
import os

app = modal.App("uploader")
volume = modal.Volume.from_name("poker-bot-checkpoints", create_if_missing=True)

CHUNK_SIZE = 50 * 1024 * 1024  # 50MB chunks

@app.function(
    volumes={"/checkpoints": volume}, 
    timeout=1800,  # 30 minutes
    memory=8192  # 8GB memory
)
def upload_checkpoint_chunk(data: bytes, filename: str, chunk_idx: int):
    """Upload a chunk of the checkpoint file."""
    import os
    
    chunk_path = f"/checkpoints/{filename}.part{chunk_idx}"
    with open(chunk_path, "wb") as f:
        f.write(data)
    
    print(f"  ✓ Uploaded chunk {chunk_idx} ({len(data) / (1024*1024):.1f} MB)")
    volume.commit()
    return True

@app.function(
    volumes={"/checkpoints": volume},
    timeout=600,
    memory=4096
)
def combine_chunks(filename: str, total_chunks: int):
    """Combine all chunks into final file."""
    import os
    
    print(f"Combining {total_chunks} chunks for {filename}...")
    output_path = f"/checkpoints/{filename}"
    
    with open(output_path, "wb") as outfile:
        for chunk_idx in range(total_chunks):
            chunk_path = f"/checkpoints/{filename}.part{chunk_idx}"
            if os.path.exists(chunk_path):
                with open(chunk_path, "rb") as infile:
                    outfile.write(infile.read())
                os.remove(chunk_path)
                print(f"  Combined chunk {chunk_idx}")
            else:
                print(f"  ⚠ Warning: chunk {chunk_idx} not found!")
    
    file_size = os.path.getsize(output_path) / (1024*1024)
    print(f"✓ Combined {filename} ({file_size:.1f} MB)")
    volume.commit()
    return True

if __name__ == "__main__":
    local_path = "checkpoints/checkpoint_iter_195.pt"
    if os.path.exists(local_path):
        file_size = os.path.getsize(local_path) / (1024*1024)
        print(f"Uploading {local_path} ({file_size:.1f} MB) to Modal...")
        print(f"Using {CHUNK_SIZE / (1024*1024):.0f} MB chunks...")
        
        chunks = []
        with open(local_path, "rb") as f:
            chunk_idx = 0
            while True:
                chunk = f.read(CHUNK_SIZE)
                if not chunk:
                    break
                chunks.append((chunk_idx, chunk))
                chunk_idx += 1
        
        total_chunks = len(chunks)
        print(f"Split into {total_chunks} chunks")
        
        with app.run():
            # Upload all chunks
            for chunk_idx, chunk_data in chunks:
                print(f"Uploading chunk {chunk_idx + 1}/{total_chunks}...")
                upload_checkpoint_chunk.remote(chunk_data, "checkpoint_iter_195.pt", chunk_idx)
            
            # Combine chunks
            print("\nCombining chunks...")
            result = combine_chunks.remote("checkpoint_iter_195.pt", total_chunks)
            
            if result:
                print("\n✓✓ Upload successful!")
            else:
                print("\n✗ Upload failed during combination!")
    else:
        print(f"✗ File {local_path} not found!")


