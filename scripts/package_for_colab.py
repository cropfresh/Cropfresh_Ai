import shutil
import os
from pathlib import Path

def package_project():
    """Package the project for Google Colab."""
    print("📦 Packaging CropFresh AI for Colab...")
    
    # Define paths
    base_dir = Path(__file__).parent.parent
    output_filename = base_dir / "notebooks" / "cropfresh_colab_package"
    
    # Create temporary directory for staging
    staging_dir = base_dir / "temp_staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir()
    
    # Directories/Files to copy
    to_copy = [
        "src",
        "data",
        ".env.example",
        "pyproject.toml",
    ]
    
    try:
        for item in to_copy:
            src_path = base_dir / item
            dst_path = staging_dir / item
            
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path)
            elif src_path.exists():
                shutil.copy2(src_path, dst_path)
            else:
                print(f"⚠️  Warning: {item} not found")
        
        # Rename .env.example to .env
        env_example = staging_dir / ".env.example"
        if env_example.exists():
            shutil.copy2(env_example, staging_dir / ".env")
            
        # Create ZIP
        shutil.make_archive(str(output_filename), 'zip', staging_dir)
        print(f"✅ Created package: {output_filename}.zip")
        print(f"   Size: {os.path.getsize(str(output_filename) + '.zip') / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        if staging_dir.exists():
            shutil.rmtree(staging_dir)

if __name__ == "__main__":
    package_project()
