import os

# Folders to be created
REQUIRED_FOLDERS = [
    'data/raw',
    'data/processed',
    'data/predictions',
    'data/weather',
    'models',
    'reports'
]

def create_folders(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f'Created or verified: {folder}')

if __name__ == '__main__':
    create_folders()
    print("✅ All necessary folders are set up.")
