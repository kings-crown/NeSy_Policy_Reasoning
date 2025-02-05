import json
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import git
import requests
from tqdm import tqdm


class URLS:
    urls: dict = {
        "minif2f": "https://github.com/openai/miniF2F.git",
        "math": "huggingface.co/datasets/EleutherAI/hendrycks_math",
    }


@dataclass
class MathQuestion:
    """A question in the MATH dataset"""

    informal_statement: str
    formal_statement: str
    informal_solution: str
    subject: str
    question_number: str
    mini_split: str | None = None
    math_split: str | None = None
    language: str | None = None


class MathDataset:
    def __init__(self, questions: list[MathQuestion]):
        self.questions = questions

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx]

    def search(self, query: str) -> list[MathQuestion]:
        """Return questions that contains the query in the informal or formal statement

        Args:
            query (str): The query to search for in the formal or informal statements.

        Returns:
            list[MathQuestion]: The list of questions that contain the query.
        """
        matches = []
        for q in self.questions:
            if query in q.informal_statement or query in q.formal_statement:
                matches.append(q)
        return matches

    def get_subject(self, subject: str) -> "MathDataset":
        """Get the dataset for a specific subject."""
        return MathDataset([q for q in self.questions if q.subject == subject])

    def get_question(self, question_number: str | int):
        """Get the dataset for a specific question from a specific subject."""
        matches = [
            q for q in self.questions if q.question_number == str(question_number)
        ]
        if len(matches) == 0:
            raise ValueError(f"No question found with number {question_number}")
        if len(matches) == 1:
            return matches[0]
        return matches


class MiniF2FMATH(MathDataset):

    """A dataset for the combined MiniF2F and MATH datasets"""

    def __init__(
        self,
        root: str | Path | None = None,
        download: bool = True,
        language: str = "isabelle",
        comments: bool = False,
    ):
        """Initialize MiniF2FMATH dataset.

        Args:
            root (str | Path | None): The root directory to store the dataset. If None,
                it will be stored in ~/.cache/autoformalism. Defaults to None.
            download (bool): Whether to download the dataset. The download will be
                skipped if the data alread exists even if this is True.  Defaults to
                True.
            language (str): The theorem prover language to use.  One of
                ["isabelle", "metamath", "hollight"]. Defaults to "isabelle".
            comments (bool): Whether to include comments in the formal statements.
                Defaults to False.
        """
        if root is None:
            root = Path.home() / ".cache/autoformalism"
        self.root = Path(root)
        self.should_download = download
        self.language = language
        self.comments = comments
        self.maybe_download()
        questions = self.load_questions()
        super().__init__(questions)

    def maybe_download(self):
        """Download the data if it does not exist and the user wants to download it"""
        if not self.should_download:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        for _, url in URLS.urls.items():
            filename = stem_from_url(url)
            if not (self.root / filename).exists():
                self.download(url)

    def download(self, url):
        download(url, self.root)

    def load_questions(self) -> list[MathQuestion]:
        """Load the aligned math questions."""
        metadata = self.load_metadata()
        questions = list(question_generator(metadata))
        if not self.comments:
            questions = [remove_comments(q, self.language) for q in questions]
        return questions

    def load_metadata(self) -> list[Tuple["FileMetadata", "FileMetadata"]]:
        """Get the aligned metadata between the MiniF2F and MATH datasets."""
        return get_aligned_metadata(self.root, language=self.language)


def remove_comments(data, language):
    if language == "isabelle":
        data.formal_statement = remove_isabelle_comments(data.formal_statement)
    if language == "metamath":
        data.formal_statement = remove_metamath_comments(data.formal_statement)
    return data


def remove_isabelle_comments(text):
    regex = re.compile(r"\(\*.*?\*\)", re.DOTALL)
    return regex.sub("", text)


def remove_metamath_comments(text):
    regex = re.compile(r"\@\(.*?\@\)", re.DOTALL)
    return regex.sub("", text)


# Data Loading Helpers
# --------------------
@dataclass
class FileMetadata:
    """Metadata for a file"""

    path: str
    subject: str
    question_number: str
    split: str | None = None
    language: str | None = None


def question_generator(aligned_metadata: list[Tuple[FileMetadata, FileMetadata]]):
    """Generates MathQuestion objects from the aligned metadata

    Args:
        aligned_metadata (list[Tuple[FileMetadata, FileMetadata]]): The aligned metadata
            between the MiniF2F and MATH datasets. Each tuple contains the metadata for
            a single MiniF2F question and the corresponding MATH question.

    Yields:
        MathQuestion: A MathQuestion object for each aligned metadata pair.

    """
    for mini, math in aligned_metadata:
        with open(mini.path) as f:
            mini_data = f.read()
        with open(math.path) as f:
            math_data = json.load(f)
        yield MathQuestion(
            **{
                "informal_statement": math_data["problem"],
                "formal_statement": mini_data,
                "informal_solution": math_data["solution"],
                "subject": mini.subject,
                "question_number": mini.question_number,
                "mini_split": mini.split,
                "math_split": math.split,
                "language": mini.language,
            }
        )


def get_aligned_metadata(
    root: str | Path, language: str
) -> list[Tuple[FileMetadata, FileMetadata]]:
    """Get the aligned dataset.

    This function scans the MiniF2F and MATH datasets for matching questions and then
    stores matching questions in a list of tuples.

    Args:
        root (str | Path): The root directory for the dataset.
        language (str): The language to use for the theorem prover.

    Returns:
        list[Tuple[FileMetadata, FileMetadata]]: The aligned metadata between the
            MiniF2F and MATH datasets.

    """
    minif2f = load_minif2f_metadata(root)
    math = load_math_metadata(root)

    dataset: list[Tuple[FileMetadata, FileMetadata]] = []
    for mini_meta in minif2f:
        for math_meta in math:
            if (
                mini_meta.subject == math_meta.subject
                and mini_meta.question_number == math_meta.question_number
            ):
                if mini_meta.language is None:
                    raise ValueError(
                        "Expected MiniF2F metadata to have a language, "
                        f"got {mini_meta} instead"
                    )
                elif mini_meta.language == language:
                    dataset.append((mini_meta, math_meta))
    return dataset


def load_minif2f_metadata(root: str | Path) -> list[FileMetadata]:
    """Load the metadata for the MiniF2F dataset.


    Args:
        root (str | Path): The root directory for the dataset.

    Returns:
        list[FileMetadata]: The metadata for the MiniF2F dataset.

    """
    root = Path(root) / "miniF2F"
    metadata = []
    math_files = root.rglob("mathd*")
    for file in math_files:
        if len(file.parts) < 3:
            raise ValueError(
                "Expected filepath to contain atleast <subject>/<split>/<filename>, "
                f"got {file.parts} instead"
            )
        lang, split = file.parts[-3], file.parts[-2]
        filename = file.stem
        split_token = "-" if "-" in filename else "_"
        file_parts = filename.split(split_token)
        if len(file_parts) != 3:
            raise ValueError(
                "Expected filename of the form mathd-<subject>-<number>, "
                f"got {filename} instead"
            )
        subject, question_number = file_parts[1], file_parts[2]
        subject = subject.casefold().replace("_", "").replace("-", "")

        metadata.append(
            FileMetadata(
                path=str(file),
                subject=subject,
                question_number=question_number,
                split=split,
                language=lang,
            )
        )

    return metadata


def load_math_metadata(root: str | Path) -> list[FileMetadata]:
    """Load the metadata for the MATH dataset.

    Args:
        root (str | Path): The root directory for the dataset.

    Returns:
        list[FileMetadata]: The metadata for the MATH dataset.

    """
    root = Path(root) / "MATH"
    metadata = []
    math_files = root.rglob("*.json")
    for file in math_files:
        if len(file.parts) < 3:
            raise ValueError(
                "Expected filepath to contain atleast <split>/<subject>/<filename>, "
                f"got {file} instead"
            )
        split, subject = file.parts[-3], file.parts[-2]
        subject = subject.casefold().replace("_", "").replace("-", "")
        question_number = file.stem

        metadata.append(
            FileMetadata(
                path=str(file),
                subject=subject,
                question_number=question_number,
                split=split,
            )
        )

    return metadata


def download(url: str, target_name: str | Path):
    """Download the URL to the target name.

    Only supports git repos and tar files currently.

    Args:
        url (str): The URL to download.
        target_name (str | Path): The target name to save the file.

    Raises:
        ValueError: If the URL is not a git or tar file.
    """
    if is_git(url):
        download_git(url, target_name)
    elif is_tar(url):
        download_and_extract_tar(url, target_name)
    else:
        raise ValueError(f"Expecting a git or tar url, got {url} instead")


def download_git(url: str, root: str | Path):
    """Download the git repository."""
    target_name = Path(root) / stem_from_url(url)
    print(f"Cloning {url} into {target_name}")
    git.Repo.clone_from(url, target_name)


def download_and_extract_tar(url: str, root: str | Path):
    """Download and extract the tar file"""
    tar_path = Path(root) / name_from_url(url)
    print(f"Downloading {url} into {tar_path}")
    download_url(url, tar_path)
    extract_tar(tar_path, root)


def download_url(url: str, target_name: str | Path):
    """Downloads the URL with a progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    with tqdm(total=total_size, unit="B", unit_scale=True) as bar:
        with open(target_name, "wb") as file:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))

    if total_size != 0 and bar.n != total_size:
        raise ValueError("Downloaded file is not the same size as the original")


def extract_tar(tar_path: str | Path, target_name: str | Path):
    """Extracts a tarfile into the directory"""
    tar_path = Path(tar_path)
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(target_name)
    tar_path.unlink()


def stem_from_url(url: str) -> str:
    return Path(url).stem


def name_from_url(url: str) -> str:
    return Path(url).name


def is_git(url: str) -> bool:
    return url.endswith(".git")


def is_tar(url: str) -> bool:
    return url.endswith(".tar.gz") or url.endswith(".tar")