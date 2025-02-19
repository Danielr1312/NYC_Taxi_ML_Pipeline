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

def create_env_file():
    env_path = ".env"

    if not os.path.exists(env_path):
        with open(env_path, 'w') as env_file:
            env_file.write("API_KEY=None\n")
        print("✅ Created .env file with API_KEY=None.")
    else:
        print(".env file already exists. No changes made.")

if __name__ == "__main__":
    create_folders(REQUIRED_FOLDERS)
    create_env_file()
    print("✅ Setup complete.")