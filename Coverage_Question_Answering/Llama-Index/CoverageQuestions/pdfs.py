import os
import PyPDF2

# Function to get file names and prepend a specific path
def get_files_with_prefixed_path(directory, prefix_path):
    file_names = []
    prefixed_paths = []
    
    # Iterate over all files in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):  # Filter for PDF files
                file_names.append(file)
                # Prefix the specified path to the filename
                prefixed_paths.append(os.path.join(prefix_path, file))
    
    return file_names, prefixed_paths

def read_pdfs(file_paths):
    pdf_contents = []
    
    for path in file_paths:
        with open(path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num in range(len(reader.pages)):
                text += reader.pages[page_num].extract_text()
            pdf_contents.append(text)
    
    return pdf_contents

# Replace 'your_directory_path' with the actual directory path you want to scan
directory_path = '/Users/samveg.shah/Desktop/Llama-index/data'
# Static prefix path as provided
prefix_path = '/Users/samveg.shah/Desktop/Llama-index/data'

file_names, prefixed_paths = get_files_with_prefixed_path(directory_path, prefix_path)
pdf_contents = read_pdfs(prefixed_paths)

# Print the results
print("File Names:", file_names)
print("Prefixed Paths:", prefixed_paths)
print("PDF Contents:", pdf_contents)