import os

# Define the folder structure
structure = {
    "data/raw": [],
    "data/processed": [],
    "data/output": [],
    "models": [],
    "notebooks": ["01_experiment.ipynb"],
    "src": [
        "__init__.py",
        "config.py",
        "detector.py",
        "analyzer.py",
        "grader.py",
        "utils.py"
    ],
    ".": ["main.py", "requirements.txt", "README.md", ".gitignore"] # Root files
}

def create_structure():
    base_path = os.getcwd()
    
    for folder, files in structure.items():
        # Create folder (handles nested paths like data/raw)
        folder_path = os.path.join(base_path, folder) if folder != "." else base_path
        os.makedirs(folder_path, exist_ok=True)
        
        # Create files inside the folder
        for file in files:
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'w') as f:
                # Add a simple docstring to python files
                if file.endswith('.py'):
                    f.write(f'"""\nModule: {file}\nDescription: [Add description here]\n"""\n')
                pass # Just creates an empty file
            print(f"Created: {file_path}")

if __name__ == "__main__":
    create_structure()
    print("\n✅ Project structure created successfully!")