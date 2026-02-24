from pathlib import Path


def repo_path_to_abs_path(path: str) -> Path:
    """
    Converts a repository-relative path to an absolute path.

    Args:
        path: A string representing the repository-relative path.

    Returns:
        A Path object representing the absolute path.

    Example usage:
        ioi_test_file = repo_path_to_abs_path("datasets/ioi/ioi_test_100.json")
    """
    repo_abs_path = Path(__file__).parent.parent.parent.absolute()
    return repo_abs_path / path
