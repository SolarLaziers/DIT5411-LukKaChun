import zipfile
import os
from pathlib import Path

# Paths
zip_path = 'all_data.zip'  # Your concatenated ZIP
output_dir = Path(r'D:\AI_Chinese_Handwrting_Recognition\cleaned_data')

# Create output dir
output_dir.mkdir(parents=True, exist_ok=True)

# Extract with Big5 encoding
print("Extracting with Big5 encoding...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for member in zip_ref.namelist():
        # Decode filename with Big5
        try:
            decoded_name = member.encode('latin1').decode('big5')  # Common fix for ZIP Big5
        except UnicodeDecodeError:
            decoded_name = member  # Fallback if pure UTF-8
        
        # Full path
        target_path = output_dir / decoded_name
        
        # Create dirs if needed
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Extract
        with zip_ref.open(member) as source, open(target_path, 'wb') as target:
            target.write(source.read())
        
        print(f"Extracted: {decoded_name}")  # Progress (optional; remove for speed)

print(f"Extraction complete! Check {output_dir}")
print("Folder count:", len([d for d in output_dir.iterdir() if d.is_dir()]))