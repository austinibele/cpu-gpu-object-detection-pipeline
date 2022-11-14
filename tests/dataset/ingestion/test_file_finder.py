from src.dataset.ingestion.file_finder import FileFinder

def test_find_files():
    file_finder = FileFinder()
    files = file_finder.find_files(dir_path="src/training", extension=".py")
    assert len(files) > 0
    assert files[0][-2:] == "py"
    
if __name__ == '__main__':
    test_find_files()
    
