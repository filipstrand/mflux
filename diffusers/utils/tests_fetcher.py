# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Welcome to tests_fetcher V2.

This util is designed to fetch tests to run on a PR so that only the tests impacted by the modifications are run, and
when too many models are being impacted, only run the tests of a subset of core models. It works like this.

Stage 1: Identify the modified files. For jobs that run on the main branch, it's just the diff with the last commit.
On a PR, this takes all the files from the branching point to the current commit (so all modifications in a PR, not
just the last commit) but excludes modifications that are on docstrings or comments only.

Stage 2: Extract the tests to run. This is done by looking at the imports in each module and test file: if module A
imports module B, then changing module B impacts module A, so the tests using module A should be run. We thus get the
dependencies of each model and then recursively builds the 'reverse' map of dependencies to get all modules and tests
impacted by a given file. We then only keep the tests (and only the core models tests if there are too many modules).

Caveats:
  - This module only filters tests by files (not individual tests) so it's better to have tests for different things
    in different files.
  - This module assumes inits are just importing things, not really building objects, so it's better to structure
    them this way and move objects building in separate submodules.

Usage:

Base use to fetch the tests in a pull request

```bash
python utils/tests_fetcher.py
```

Base use to fetch the tests on a the main branch (with diff from the last commit):

```bash
python utils/tests_fetcher.py --diff_with_last_commit
```
"""

import argparse
import collections
import json
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from git import Repo


PATH_TO_REPO = Path(__file__).parent.parent.resolve()
PATH_TO_EXAMPLES = PATH_TO_REPO / "examples"
PATH_TO_DIFFUSERS = PATH_TO_REPO / "src/diffusers"
PATH_TO_TESTS = PATH_TO_REPO / "tests"

# Ignore fixtures in tests folder
# Ignore lora since they are always tested
MODULES_TO_IGNORE = ["fixtures", "lora"]

IMPORTANT_PIPELINES = [
    "controlnet",
    "stable_diffusion",
    "stable_diffusion_2",
    "stable_diffusion_xl",
    "stable_video_diffusion",
    "deepfloyd_if",
    "kandinsky",
    "kandinsky2_2",
    "text_to_video_synthesis",
    "wuerstchen",
]


@contextmanager
def checkout_commit(repo: Repo, commit_id: str):
    """
    Context manager that checks out a given commit when entered, but gets back to the reference it was at on exit.

    Args:
        repo (`git.Repo`): A git repository (for instance the Transformers repo).
        commit_id (`str`): The commit reference to checkout inside the context manager.
    """
    current_head = repo.head.commit if repo.head.is_detached else repo.head.ref

    try:
        repo.git.checkout(commit_id)
        yield

    finally:
        repo.git.checkout(current_head)


def clean_code(content: str) -> str:
    """
    Remove docstrings, empty line or comments from some code (used to detect if a diff is real or only concern
    comments or docstrings).

    Args:
        content (`str`): The code to clean

    Returns:
        `str`: The cleaned code.
    """
    # We need to deactivate autoformatting here to write escaped triple quotes (we cannot use real triple quotes or
    # this would mess up the result if this function applied to this particular file).
    # fmt: off
    # Remove docstrings by splitting on triple " then triple ':
    splits = content.split('\"\"\"')
    content = "".join(splits[::2])
    splits = content.split("\'\'\'")
    # fmt: on
    content = "".join(splits[::2])

    # Remove empty lines and comments
    lines_to_keep = []
    for line in content.split("\n"):
        # remove anything that is after a # sign.
        line = re.sub("#.*$", "", line)
        # remove white lines
        if len(line) != 0 and not line.isspace():
            lines_to_keep.append(line)
    return "\n".join(lines_to_keep)


def keep_doc_examples_only(content: str) -> str:
    """
    Remove everything from the code content except the doc examples (used to determined if a diff should trigger doc
    tests or not).

    Args:
        content (`str`): The code to clean

    Returns:
        `str`: The cleaned code.
    """
    # Keep doc examples only by splitting on triple "`"
    splits = content.split("```")
    # Add leading and trailing "```" so the navigation is easier when compared to the original input `content`
    content = "```" + "```".join(splits[1::2]) + "```"

    # Remove empty lines and comments
    lines_to_keep = []
    for line in content.split("\n"):
        # remove anything that is after a # sign.
        line = re.sub("#.*$", "", line)
        # remove white lines
        if len(line) != 0 and not line.isspace():
            lines_to_keep.append(line)
    return "\n".join(lines_to_keep)


def get_all_tests() -> List[str]:
    """
    Walks the `tests` folder to return a list of files/subfolders. This is used to split the tests to run when using
    parallelism. The split is:

    - folders under `tests`: (`tokenization`, `pipelines`, etc) except the subfolder `models` is excluded.
    - folders under `tests/models`: `bert`, `gpt2`, etc.
    - test files under `tests`: `test_modeling_common.py`, `test_tokenization_common.py`, etc.
    """

    # test folders/files directly under `tests` folder
    tests = os.listdir(PATH_TO_TESTS)
    tests = [f"tests/{f}" for f in tests if "__pycache__" not in f]
    tests = sorted([f for f in tests if (PATH_TO_REPO / f).is_dir() or f.startswith("tests/test_")])

    return tests


def diff_is_docstring_only(repo: Repo, branching_point: str, filename: str) -> bool:
    """
    Check if the diff is only in docstrings (or comments and whitespace) in a filename.

    Args:
        repo (`git.Repo`): A git repository (for instance the Transformers repo).
        branching_point (`str`): The commit reference of where to compare for the diff.
        filename (`str`): The filename where we want to know if the diff isonly in docstrings/comments.

    Returns:
        `bool`: Whether the diff is docstring/comments only or not.
    """
    folder = Path(repo.working_dir)
    with checkout_commit(repo, branching_point):
        with open(folder / filename, "r", encoding="utf-8") as f:
            old_content = f.read()

    with open(folder / filename, "r", encoding="utf-8") as f:
        new_content = f.read()

    old_content_clean = clean_code(old_content)
    new_content_clean = clean_code(new_content)

    return old_content_clean == new_content_clean


def diff_contains_doc_examples(repo: Repo, branching_point: str, filename: str) -> bool:
    """
    Check if the diff is only in code examples of the doc in a filename.

    Args:
        repo (`git.Repo`): A git repository (for instance the Transformers repo).
        branching_point (`str`): The commit reference of where to compare for the diff.
        filename (`str`): The filename where we want to know if the diff is only in codes examples.

    Returns:
        `bool`: Whether the diff is only in code examples of the doc or not.
    """
    folder = Path(repo.working_dir)
    with checkout_commit(repo, branching_point):
        with open(folder / filename, "r", encoding="utf-8") as f:
            old_content = f.read()

    with open(folder / filename, "r", encoding="utf-8") as f:
        new_content = f.read()

    old_content_clean = keep_doc_examples_only(old_content)
    new_content_clean = keep_doc_examples_only(new_content)

    return old_content_clean != new_content_clean


def get_diff(repo: Repo, base_commit: str, commits: List[str]) -> List[str]:
    """
    Get the diff between a base commit and one or several commits.

    Args:
        repo (`git.Repo`):
            A git repository (for instance the Transformers repo).
        base_commit (`str`):
            The commit reference of where to compare for the diff. This is the current commit, not the branching point!
        commits (`List[str]`):
            The list of commits with which to compare the repo at `base_commit` (so the branching point).

    Returns:
        `List[str]`: The list of Python files with a diff (files added, renamed or deleted are always returned, files
        modified are returned if the diff in the file is not only in docstrings or comments, see
        `diff_is_docstring_only`).
    """
    print("\n### DIFF ###\n")
    code_diff = []
    for commit in commits:
        for diff_obj in commit.diff(base_commit):
            # We always add new python files
            if diff_obj.change_type == "A" and diff_obj.b_path.endswith(".py"):
                code_diff.append(diff_obj.b_path)
            # We check that deleted python files won't break corresponding tests.
            elif diff_obj.change_type == "D" and diff_obj.a_path.endswith(".py"):
                code_diff.append(diff_obj.a_path)
            # Now for modified files
            elif diff_obj.change_type in ["M", "R"] and diff_obj.b_path.endswith(".py"):
                # In case of renames, we'll look at the tests using both the old and new name.
                if diff_obj.a_path != diff_obj.b_path:
                    code_diff.extend([diff_obj.a_path, diff_obj.b_path])
                else:
                    # Otherwise, we check modifications are in code and not docstrings.
                    if diff_is_docstring_only(repo, commit, diff_obj.b_path):
                        print(f"Ignoring diff in {diff_obj.b_path} as it only concerns docstrings or comments.")
                    else:
                        code_diff.append(diff_obj.a_path)

    return code_diff


def get_modified_python_files(diff_with_last_commit: bool = False) -> List[str]:
    """
    Return a list of python files that have been modified between:

    - the current head and the main branch if `diff_with_last_commit=False` (default)
    - the current head and its parent commit otherwise.

    Returns:
        `List[str]`: The list of Python files with a diff (files added, renamed or deleted are always returned, files
        modified are returned if the diff in the file is not only in docstrings or comments, see
        `diff_is_docstring_only`).
    """
    repo = Repo(PATH_TO_REPO)

    if not diff_with_last_commit:
        # Need to fetch refs for main using remotes when running with github actions.
        upstream_main = repo.remotes.origin.refs.main

        print(f"main is at {upstream_main.commit}")
        print(f"Current head is at {repo.head.commit}")

        branching_commits = repo.merge_base(upstream_main, repo.head)
        for commit in branching_commits:
            print(f"Branching commit: {commit}")
        return get_diff(repo, repo.head.commit, branching_commits)
    else:
        print(f"main is at {repo.head.commit}")
        parent_commits = repo.head.commit.parents
        for commit in parent_commits:
            print(f"Parent commit: {commit}")
        return get_diff(repo, repo.head.commit, parent_commits)


def get_diff_for_doctesting(repo: Repo, base_commit: str, commits: List[str]) -> List[str]:
    """
    Get the diff in doc examples between a base commit and one or several commits.

    Args:
        repo (`git.Repo`):
            A git repository (for instance the Transformers repo).
        base_commit (`str`):
            The commit reference of where to compare for the diff. This is the current commit, not the branching point!
        commits (`List[str]`):
            The list of commits with which to compare the repo at `base_commit` (so the branching point).

    Returns:
        `List[str]`: The list of Python and Markdown files with a diff (files added or renamed are always returned, files
        modified are returned if the diff in the file is only in doctest examples).
    """
    print("\n### DIFF ###\n")
    code_diff = []
    for commit in commits:
        for diff_obj in commit.diff(base_commit):
            # We only consider Python files and doc files.
            if not diff_obj.b_path.endswith(".py") and not diff_obj.b_path.endswith(".md"):
                continue
            # We always add new python/md files
            if diff_obj.change_type in ["A"]:
                code_diff.append(diff_obj.b_path)
            # Now for modified files
            elif diff_obj.change_type in ["M", "R"]:
                # In case of renames, we'll look at the tests using both the old and new name.
                if diff_obj.a_path != diff_obj.b_path:
                    code_diff.extend([diff_obj.a_path, diff_obj.b_path])
                else:
                    # Otherwise, we check modifications contain some doc example(s).
                    if diff_contains_doc_examples(repo, commit, diff_obj.b_path):
                        code_diff.append(diff_obj.a_path)
                    else:
                        print(f"Ignoring diff in {diff_obj.b_path} as it doesn't contain any doc example.")

    return code_diff


def get_all_doctest_files() -> List[str]:
    """
    Return the complete list of python and Markdown files on which we run doctest.

    At this moment, we restrict this to only take files from `src/` or `docs/source/en/` that are not in `utils/not_doctested.txt`.

    Returns:
        `List[str]`: The complete list of Python and Markdown files on which we run doctest.
    """
    py_files = [str(x.relative_to(PATH_TO_REPO)) for x in PATH_TO_REPO.glob("**/*.py")]
    md_files = [str(x.relative_to(PATH_TO_REPO)) for x in PATH_TO_REPO.glob("**/*.md")]
    test_files_to_run = py_files + md_files

    # only include files in `src` or `docs/source/en/`
    test_files_to_run = [x for x in test_files_to_run if x.startswith(("src/", "docs/source/en/"))]
    # not include init files
    test_files_to_run = [x for x in test_files_to_run if not x.endswith(("__init__.py",))]

    # These are files not doctested yet.
    with open("utils/not_doctested.txt") as fp:
        not_doctested = {x.split(" ")[0] for x in fp.read().strip().split("\n")}

    # So far we don't have 100% coverage for doctest. This line will be removed once we achieve 100%.
    test_files_to_run = [x for x in test_files_to_run if x not in not_doctested]

    return sorted(test_files_to_run)


def get_new_doctest_files(repo, base_commit, branching_commit) -> List[str]:
    """
    Get the list of files that were removed from "utils/not_doctested.txt", between `base_commit` and
    `branching_commit`.

    Returns:
        `List[str]`: List of files that were removed from "utils/not_doctested.txt".
    """
    for diff_obj in branching_commit.diff(base_commit):
        # Ignores all but the "utils/not_doctested.txt" file.
        if diff_obj.a_path != "utils/not_doctested.txt":
            continue
        # Loads the two versions
        folder = Path(repo.working_dir)
        with checkout_commit(repo, branching_commit):
            with open(folder / "utils/not_doctested.txt", "r", encoding="utf-8") as f:
                old_content = f.read()
        with open(folder / "utils/not_doctested.txt", "r", encoding="utf-8") as f:
            new_content = f.read()
        # Compute the removed lines and return them
        removed_content = {x.split(" ")[0] for x in old_content.split("\n")} - {
            x.split(" ")[0] for x in new_content.split("\n")
        }
        return sorted(removed_content)
    return []


def get_doctest_files(diff_with_last_commit: bool = False) -> List[str]:
    """
    Return a list of python and Markdown files where doc example have been modified between:

    - the current head and the main branch if `diff_with_last_commit=False` (default)
    - the current head and its parent commit otherwise.

    Returns:
        `List[str]`: The list of Python and Markdown files with a diff (files added or renamed are always returned, files
        modified are returned if the diff in the file is only in doctest examples).
    """
    repo = Repo(PATH_TO_REPO)

    test_files_to_run = []  # noqa
    if not diff_with_last_commit:
        upstream_main = repo.remotes.origin.refs.main
        print(f"main is at {upstream_main.commit}")
        print(f"Current head is at {repo.head.commit}")

        branching_commits = repo.merge_base(upstream_main, repo.head)
        for commit in branching_commits:
            print(f"Branching commit: {commit}")
        test_files_to_run = get_diff_for_doctesting(repo, repo.head.commit, branching_commits)
    else:
        print(f"main is at {repo.head.commit}")
        parent_commits = repo.head.commit.parents
        for commit in parent_commits:
            print(f"Parent commit: {commit}")
        test_files_to_run = get_diff_for_doctesting(repo, repo.head.commit, parent_commits)

    all_test_files_to_run = get_all_doctest_files()

    # Add to the test files to run any removed entry from "utils/not_doctested.txt".
    new_test_files = get_new_doctest_files(repo, repo.head.commit, upstream_main.commit)
    test_files_to_run = list(set(test_files_to_run + new_test_files))

    # Do not run slow doctest tests on CircleCI
    with open("utils/slow_documentation_tests.txt") as fp:
        slow_documentation_tests = set(fp.read().strip().split("\n"))
    test_files_to_run = [
        x for x in test_files_to_run if x in all_test_files_to_run and x not in slow_documentation_tests
    ]

    # Make sure we did not end up with a test file that was removed
    test_files_to_run = [f for f in test_files_to_run if (PATH_TO_REPO / f).exists()]

    return sorted(test_files_to_run)


# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+(\.+\S+)\s+import\s+([^\n]+) -> Line only contains from .xxx import yyy and we catch .xxx and yyy
# (?=\n) -> Look-ahead to a new line. We can't just put \n here or using find_all on this re will only catch every
#           other import.
_re_single_line_relative_imports = re.compile(r"(?:^|\n)\s*from\s+(\.+\S+)\s+import\s+([^\n]+)(?=\n)")
# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+(\.+\S+)\s+import\s+\(([^\)]+)\) -> Line continues with from .xxx import (yyy) and we catch .xxx and yyy
# yyy will take multiple lines otherwise there wouldn't be parenthesis.
_re_multi_line_relative_imports = re.compile(r"(?:^|\n)\s*from\s+(\.+\S+)\s+import\s+\(([^\)]+)\)")
# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+transformers(\S*)\s+import\s+([^\n]+) -> Line only contains from transformers.xxx import yyy and we catch
#           .xxx and yyy
# (?=\n) -> Look-ahead to a new line. We can't just put \n here or using find_all on this re will only catch every
#           other import.
_re_single_line_direct_imports = re.compile(r"(?:^|\n)\s*from\s+diffusers(\S*)\s+import\s+([^\n]+)(?=\n)")
# (:?^|\n) -> Non-catching group for the beginning of the doc or a new line.
# \s*from\s+transformers(\S*)\s+import\s+\(([^\)]+)\) -> Line continues with from transformers.xxx import (yyy) and we
# catch .xxx and yyy. yyy will take multiple lines otherwise there wouldn't be parenthesis.
_re_multi_line_direct_imports = re.compile(r"(?:^|\n)\s*from\s+diffusers(\S*)\s+import\s+\(([^\)]+)\)")


def extract_imports(module_fname: str, cache: Dict[str, List[str]] = None) -> List[str]:
    """
    Get the imports a given module makes.

    Args:
        module_fname (`str`):
            The name of the file of the module where we want to look at the imports (given relative to the root of
            the repo).
        cache (Dictionary `str` to `List[str]`, *optional*):
            To speed up this function if it was previously called on `module_fname`, the cache of all previously
            computed results.

    Returns:
        `List[str]`: The list of module filenames imported in the input `module_fname` (a submodule we import from that
        is a subfolder will give its init file).
    """
    if cache is not None and module_fname in cache:
        return cache[module_fname]

    with open(PATH_TO_REPO / module_fname, "r", encoding="utf-8") as f:
        content = f.read()

    # Filter out all docstrings to not get imports in code examples. As before we need to deactivate formatting to
    # keep this as escaped quotes and avoid this function failing on this file.
    # fmt: off
    splits = content.split('\"\"\"')
    # fmt: on
    content = "".join(splits[::2])

    module_parts = str(module_fname).split(os.path.sep)
    imported_modules = []

    # Let's start with relative imports
    relative_imports = _re_single_line_relative_imports.findall(content)
    relative_imports = [
        (mod, imp) for mod, imp in relative_imports if "# tests_ignore" not in imp and imp.strip() != "("
    ]
    multiline_relative_imports = _re_multi_line_relative_imports.findall(content)
    relative_imports += [(mod, imp) for mod, imp in multiline_relative_imports if "# tests_ignore" not in imp]

    # We need to remove parts of the module name depending on the depth of the relative imports.
    for module, imports in relative_imports:
        level = 0
        while module.startswith("."):
            module = module[1:]
            level += 1

        if len(module) > 0:
            dep_parts = module_parts[: len(module_parts) - level] + module.split(".")
        else:
            dep_parts = module_parts[: len(module_parts) - level]
        imported_module = os.path.sep.join(dep_parts)
        imported_modules.append((imported_module, [imp.strip() for imp in imports.split(",")]))

    # Let's continue with direct imports
    direct_imports = _re_single_line_direct_imports.findall(content)
    direct_imports = [(mod, imp) for mod, imp in direct_imports if "# tests_ignore" not in imp and imp.strip() != "("]
    multiline_direct_imports = _re_multi_line_direct_imports.findall(content)
    direct_imports += [(mod, imp) for mod, imp in multiline_direct_imports if "# tests_ignore" not in imp]

    # We need to find the relative path of those imports.
    for module, imports in direct_imports:
        import_parts = module.split(".")[1:]  # ignore the name of the repo since we add it below.
        dep_parts = ["src", "diffusers"] + import_parts
        imported_module = os.path.sep.join(dep_parts)
        imported_modules.append((imported_module, [imp.strip() for imp in imports.split(",")]))

    result = []
    # Double check we get proper modules (either a python file or a folder with an init).
    for module_file, imports in imported_modules:
        if (PATH_TO_REPO / f"{module_file}.py").is_file():
            module_file = f"{module_file}.py"
        elif (PATH_TO_REPO / module_file).is_dir() and (PATH_TO_REPO / module_file / "__init__.py").is_file():
            module_file = os.path.sep.join([module_file, "__init__.py"])
        imports = [imp for imp in imports if len(imp) > 0 and re.match("^[A-Za-z0-9_]*$", imp)]
        if len(imports) > 0:
            result.append((module_file, imports))

    if cache is not None:
        cache[module_fname] = result

    return result


def get_module_dependencies(module_fname: str, cache: Dict[str, List[str]] = None) -> List[str]:
    """
    Refines the result of `extract_imports` to remove subfolders and get a proper list of module filenames: if a file
    as an import `from utils import Foo, Bar`, with `utils` being a subfolder containing many files, this will traverse
    the `utils` init file to check where those dependencies come from: for instance the files utils/foo.py and utils/bar.py.

    Warning: This presupposes that all intermediate inits are properly built (with imports from the respective
    submodules) and work better if objects are defined in submodules and not the intermediate init (otherwise the
    intermediate init is added, and inits usually have a lot of dependencies).

    Args:
        module_fname (`str`):
            The name of the file of the module where we want to look at the imports (given relative to the root of
            the repo).
        cache (Dictionary `str` to `List[str]`, *optional*):
            To speed up this function if it was previously called on `module_fname`, the cache of all previously
            computed results.

    Returns:
        `List[str]`: The list of module filenames imported in the input `module_fname` (with submodule imports refined).
    """
    dependencies = []
    imported_modules = extract_imports(module_fname, cache=cache)
    # The while loop is to recursively traverse all inits we may encounter: we will add things as we go.
    while len(imported_modules) > 0:
        new_modules = []
        for module, imports in imported_modules:
            # If we end up in an __init__ we are often not actually importing from this init (except in the case where
            # the object is fully defined in the __init__)
            if module.endswith("__init__.py"):
                # So we get the imports from that init then try to find where our objects come from.
                new_imported_modules = extract_imports(module, cache=cache)
                for new_module, new_imports in new_imported_modules:
                    if any(i in new_imports for i in imports):
                        if new_module not in dependencies:
                            new_modules.append((new_module, [i for i in new_imports if i in imports]))
                        imports = [i for i in imports if i not in new_imports]
                if len(imports) > 0:
                    # If there are any objects lefts, they may be a submodule
                    path_to_module = PATH_TO_REPO / module.replace("__init__.py", "")
                    dependencies.extend(
                        [
                            os.path.join(module.replace("__init__.py", ""), f"{i}.py")
                            for i in imports
                            if (path_to_module / f"{i}.py").is_file()
                        ]
                    )
                    imports = [i for i in imports if not (path_to_module / f"{i}.py").is_file()]
                    if len(imports) > 0:
                        # Then if there are still objects left, they are fully defined in the init, so we keep it as a
                        # dependency.
                        dependencies.append(module)
            else:
                dependencies.append(module)

        imported_modules = new_modules

    return dependencies


def create_reverse_dependency_tree() -> List[Tuple[str, str]]:
    """
    Create a list of all edges (a, b) which mean that modifying a impacts b with a going over all module and test files.
    """
    cache = {}
    all_modules = list(PATH_TO_DIFFUSERS.glob("**/*.py")) + list(PATH_TO_TESTS.glob("**/*.py"))
    all_modules = [str(mod.relative_to(PATH_TO_REPO)) for mod in all_modules]
    edges = [(dep, mod) for mod in all_modules for dep in get_module_dependencies(mod, cache=cache)]

    return list(set(edges))


def get_tree_starting_at(module: str, edges: List[Tuple[str, str]]) -> List[Union[str, List[str]]]:
    """
    Returns the tree starting at a given module following all edges.

    Args:
        module (`str`): The module that will be the root of the subtree we want.
        edges (`List[Tuple[str, str]]`): The list of all edges of the tree.

    Returns:
        `List[Union[str, List[str]]]`: The tree to print in the following format: [module, [list of edges
        starting at module], [list of edges starting at the preceding level], ...]
    """
    vertices_seen = [module]
    new_edges = [edge for edge in edges if edge[0] == module and edge[1] != module and "__init__.py" not in edge[1]]
    tree = [module]
    while len(new_edges) > 0:
        tree.append(new_edges)
        final_vertices = list({edge[1] for edge in new_edges})
        vertices_seen.extend(final_vertices)
        new_edges = [
            edge
            for edge in edges
            if edge[0] in final_vertices and edge[1] not in vertices_seen and "__init__.py" not in edge[1]
        ]

    return tree


def print_tree_deps_of(module, all_edges=None):
    """
    Prints the tree of modules depending on a given module.

    Args:
        module (`str`): The module that will be the root of the subtree we want.
        all_edges (`List[Tuple[str, str]]`, *optional*):
            The list of all edges of the tree. Will be set to `create_reverse_dependency_tree()` if not passed.
    """
    if all_edges is None:
        all_edges = create_reverse_dependency_tree()
    tree = get_tree_starting_at(module, all_edges)

    # The list of lines is a list of tuples (line_to_be_printed, module)
    # Keeping the modules lets us know where to insert each new lines in the list.
    lines = [(tree[0], tree[0])]
    for index in range(1, len(tree)):
        edges = tree[index]
        start_edges = {edge[0] for edge in edges}

        for start in start_edges:
            end_edges = {edge[1] for edge in edges if edge[0] == start}
            # We will insert all those edges just after the line showing start.
            pos = 0
            while lines[pos][1] != start:
                pos += 1
            lines = lines[: pos + 1] + [(" " * (2 * index) + end, end) for end in end_edges] + lines[pos + 1 :]

    for line in lines:
        # We don't print the refs that where just here to help build lines.
        print(line[0])


def init_test_examples_dependencies() -> Tuple[Dict[str, List[str]], List[str]]:
    """
    The test examples do not import from the examples (which are just scripts, not modules) so we need some extra
    care initializing the dependency map, which is the goal of this function. It initializes the dependency map for
    example files by linking each example to the example test file for the example framework.

    Returns:
        `Tuple[Dict[str, List[str]], List[str]]`: A tuple with two elements: the initialized dependency map which is a
        dict test example file to list of example files potentially tested by that test file, and the list of all
        example files (to avoid recomputing it later).
    """
    test_example_deps = {}
    all_examples = []
    for framework in ["flax", "pytorch", "tensorflow"]:
        test_files = list((PATH_TO_EXAMPLES / framework).glob("test_*.py"))
        all_examples.extend(test_files)
        # Remove the files at the root of examples/framework since they are not proper examples (they are either utils
        # or example test files).
        examples = [
            f for f in (PATH_TO_EXAMPLES / framework).glob("**/*.py") if f.parent != PATH_TO_EXAMPLES / framework
        ]
        all_examples.extend(examples)
        for test_file in test_files:
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
            # Map all examples to the test files found in examples/framework.
            test_example_deps[str(test_file.relative_to(PATH_TO_REPO))] = [
                str(e.relative_to(PATH_TO_REPO)) for e in examples if e.name in content
            ]
            # Also map the test files to themselves.
            test_example_deps[str(test_file.relative_to(PATH_TO_REPO))].append(
                str(test_file.relative_to(PATH_TO_REPO))
            )
    return test_example_deps, all_examples


def create_reverse_dependency_map() -> Dict[str, List[str]]:
    """
    Create the dependency map from module/test filename to the list of modules/tests that depend on it recursively.

    Returns:
        `Dict[str, List[str]]`: The reverse dependency map as a dictionary mapping filenames to all the filenames
        depending on it recursively. This way the tests impacted by a change in file A are the test files in the list
        corresponding to key A in this result.
    """
    cache = {}
    # Start from the example deps init.
    example_deps, examples = init_test_examples_dependencies()
    # Add all modules and all tests to all examples
    all_modules = list(PATH_TO_DIFFUSERS.glob("**/*.py")) + list(PATH_TO_TESTS.glob("**/*.py")) + examples
    all_modules = [str(mod.relative_to(PATH_TO_REPO)) for mod in all_modules]
    # Compute the direct dependencies of all modules.
    direct_deps = {m: get_module_dependencies(m, cache=cache) for m in all_modules}
    direct_deps.update(example_deps)

    # This recurses the dependencies
    something_changed = True
    while something_changed:
        something_changed = False
        for m in all_modules:
            for d in direct_deps[m]:
                # We stop recursing at an init (cause we always end up in the main init and we don't want to add all
                # files which the main init imports)
                if d.endswith("__init__.py"):
                    continue
                if d not in direct_deps:
                    raise ValueError(f"KeyError:{d}. From {m}")
                new_deps = set(direct_deps[d]) - set(direct_deps[m])
                if len(new_deps) > 0:
                    direct_deps[m].extend(list(new_deps))
                    something_changed = True

    # Finally we can build the reverse map.
    reverse_map = collections.defaultdict(list)
    for m in all_modules:
        for d in direct_deps[m]:
            reverse_map[d].append(m)

    # For inits, we don't do the reverse deps but the direct deps: if modifying an init, we want to make sure we test
    # all the modules impacted by that init.
    for m in [f for f in all_modules if f.endswith("__init__.py")]:
        direct_deps = get_module_dependencies(m, cache=cache)
        deps = sum([reverse_map[d] for d in direct_deps if not d.endswith("__init__.py")], direct_deps)
        reverse_map[m] = list(set(deps) - {m})

    return reverse_map


def create_module_to_test_map(reverse_map: Dict[str, List[str]] = None) -> Dict[str, List[str]]:
    """
    Extract the tests from the reverse_dependency_map and potentially filters the model tests.

    Args:
        reverse_map (`Dict[str, List[str]]`, *optional*):
            The reverse dependency map as created by `create_reverse_dependency_map`. Will default to the result of
            that function if not provided.
        filter_pipelines (`bool`, *optional*, defaults to `False`):
            Whether or not to filter pipeline tests to only include core pipelines if a file impacts a lot of models.

    Returns:
        `Dict[str, List[str]]`: A dictionary that maps each file to the tests to execute if that file was modified.
    """
    if reverse_map is None:
        reverse_map = create_reverse_dependency_map()

    # Utility that tells us if a given file is a test (taking test examples into account)
    def is_test(fname):
        if fname.startswith("tests"):
            return True
        if fname.startswith("examples") and fname.split(os.path.sep)[-1].startswith("test"):
            return True
        return False

    # Build the test map
    test_map = {module: [f for f in deps if is_test(f)] for module, deps in reverse_map.items()}

    return test_map


def check_imports_all_exist():
    """
    Isn't used per se by the test fetcher but might be used later as a quality check. Putting this here for now so the
    code is not lost. This checks all imports in a given file do exist.
    """
    cache = {}
    all_modules = list(PATH_TO_DIFFUSERS.glob("**/*.py")) + list(PATH_TO_TESTS.glob("**/*.py"))
    all_modules = [str(mod.relative_to(PATH_TO_REPO)) for mod in all_modules]
    direct_deps = {m: get_module_dependencies(m, cache=cache) for m in all_modules}

    for module, deps in direct_deps.items():
        for dep in deps:
            if not (PATH_TO_REPO / dep).is_file():
                print(f"{module} has dependency on {dep} which does not exist.")


def _print_list(l) -> str:
    """
    Pretty print a list of elements with one line per element and a - starting each line.
    """
    return "\n".join([f"- {f}" for f in l])


def update_test_map_with_core_pipelines(json_output_file: str):
    print(f"\n### ADD CORE PIPELINE TESTS ###\n{_print_list(IMPORTANT_PIPELINES)}")
    with open(json_output_file, "rb") as fp:
        test_map = json.load(fp)

    # Add core pipelines as their own test group
    test_map["core_pipelines"] = " ".join(
        sorted([str(PATH_TO_TESTS / f"pipelines/{pipe}") for pipe in IMPORTANT_PIPELINES])
    )

    # If there are no existing pipeline tests save the map
    if "pipelines" not in test_map:
        with open(json_output_file, "w", encoding="UTF-8") as fp:
            json.dump(test_map, fp, ensure_ascii=False)

    pipeline_tests = test_map.pop("pipelines")
    pipeline_tests = pipeline_tests.split(" ")

    # Remove core pipeline tests from the fetched pipeline tests
    updated_pipeline_tests = []
    for pipe in pipeline_tests:
        if pipe == "tests/pipelines" or Path(pipe).parts[2] in IMPORTANT_PIPELINES:
            continue
        updated_pipeline_tests.append(pipe)

    if len(updated_pipeline_tests) > 0:
        test_map["pipelines"] = " ".join(sorted(updated_pipeline_tests))

    with open(json_output_file, "w", encoding="UTF-8") as fp:
        json.dump(test_map, fp, ensure_ascii=False)


def create_json_map(test_files_to_run: List[str], json_output_file: Optional[str] = None):
    """
    Creates a map from a list of tests to run to easily split them by category, when running parallelism of slow tests.

    Args:
        test_files_to_run (`List[str]`): The list of tests to run.
        json_output_file (`str`): The path where to store the built json map.
    """
    if json_output_file is None:
        return

    test_map = {}
    for test_file in test_files_to_run:
        # `test_file` is a path to a test folder/file, starting with `tests/`. For example,
        #   - `tests/models/bert/test_modeling_bert.py` or `tests/models/bert`
        #   - `tests/trainer/test_trainer.py` or `tests/trainer`
        #   - `tests/test_modeling_common.py`
        names = test_file.split(os.path.sep)
        module = names[1]
        if module in MODULES_TO_IGNORE:
            continue

        if len(names) > 2 or not test_file.endswith(".py"):
            # test folders under `tests` or python files under them
            # take the part like tokenization, `pipeline`, etc. for other test categories
            key = os.path.sep.join(names[1:2])
        else:
            # common test files directly under `tests/`
            key = "common"

        if key not in test_map:
            test_map[key] = []
        test_map[key].append(test_file)

    # sort the keys & values
    keys = sorted(test_map.keys())
    test_map = {k: " ".join(sorted(test_map[k])) for k in keys}

    with open(json_output_file, "w", encoding="UTF-8") as fp:
        json.dump(test_map, fp, ensure_ascii=False)


def infer_tests_to_run(
    output_file: str,
    diff_with_last_commit: bool = False,
    json_output_file: Optional[str] = None,
):
    """
    The main function called by the test fetcher. Determines the tests to run from the diff.

    Args:
        output_file (`str`):
            The path where to store the summary of the test fetcher analysis. Other files will be stored in the same
            folder:

            - examples_test_list.txt: The list of examples tests to run.
            - test_repo_utils.txt: Will indicate if the repo utils tests should be run or not.
            - doctest_list.txt: The list of doctests to run.

        diff_with_last_commit (`bool`, *optional*, defaults to `False`):
            Whether to analyze the diff with the last commit (for use on the main branch after a PR is merged) or with
            the branching point from main (for use on each PR).
        filter_models (`bool`, *optional*, defaults to `True`):
            Whether or not to filter the tests to core models only, when a file modified results in a lot of model
            tests.
        json_output_file (`str`, *optional*):
            The path where to store the json file mapping categories of tests to tests to run (used for parallelism or
            the slow tests).
    """
    modified_files = get_modified_python_files(diff_with_last_commit=diff_with_last_commit)
    print(f"\n### MODIFIED FILES ###\n{_print_list(modified_files)}")
    # Create the map that will give us all impacted modules.
    reverse_map = create_reverse_dependency_map()
    impacted_files = modified_files.copy()
    for f in modified_files:
        if f in reverse_map:
            impacted_files.extend(reverse_map[f])

    # Remove duplicates
    impacted_files = sorted(set(impacted_files))
    print(f"\n### IMPACTED FILES ###\n{_print_list(impacted_files)}")

    # Grab the corresponding test files:
    if any(x in modified_files for x in ["setup.py"]):
        test_files_to_run = ["tests", "examples"]

    # in order to trigger pipeline tests even if no code change at all
    if "tests/utils/tiny_model_summary.json" in modified_files:
        test_files_to_run = ["tests"]
        any(f.split(os.path.sep)[0] == "utils" for f in modified_files)
    else:
        # All modified tests need to be run.
        test_files_to_run = [
            f for f in modified_files if f.startswith("tests") and f.split(os.path.sep)[-1].startswith("test")
        ]
        # Then we grab the corresponding test files.
        test_map = create_module_to_test_map(reverse_map=reverse_map)
        for f in modified_files:
            if f in test_map:
                test_files_to_run.extend(test_map[f])
        test_files_to_run = sorted(set(test_files_to_run))
        # Make sure we did not end up with a test file that was removed
        test_files_to_run = [f for f in test_files_to_run if (PATH_TO_REPO / f).exists()]

        any(f.split(os.path.sep)[0] == "utils" for f in modified_files)

    examples_tests_to_run = [f for f in test_files_to_run if f.startswith("examples")]
    test_files_to_run = [f for f in test_files_to_run if not f.startswith("examples")]
    print(f"\n### TEST TO RUN ###\n{_print_list(test_files_to_run)}")
    if len(test_files_to_run) > 0:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(" ".join(test_files_to_run))

        # Create a map that maps test categories to test files, i.e. `models/bert` -> [...test_modeling_bert.py, ...]

        # Get all test directories (and some common test files) under `tests` and `tests/models` if `test_files_to_run`
        # contains `tests` (i.e. when `setup.py` is changed).
        if "tests" in test_files_to_run:
            test_files_to_run = get_all_tests()

        create_json_map(test_files_to_run, json_output_file)

    print(f"\n### EXAMPLES TEST TO RUN ###\n{_print_list(examples_tests_to_run)}")
    if len(examples_tests_to_run) > 0:
        # We use `all` in the case `commit_flags["test_all"]` as well as in `create_circleci_config.py` for processing
        if examples_tests_to_run == ["examples"]:
            examples_tests_to_run = ["all"]
        example_file = Path(output_file).parent / "examples_test_list.txt"
        with open(example_file, "w", encoding="utf-8") as f:
            f.write(" ".join(examples_tests_to_run))


def filter_tests(output_file: str, filters: List[str]):
    """
    Reads the content of the output file and filters out all the tests in a list of given folders.

    Args:
        output_file (`str` or `os.PathLike`): The path to the output file of the tests fetcher.
        filters (`List[str]`): A list of folders to filter.
    """
    if not os.path.isfile(output_file):
        print("No test file found.")
        return
    with open(output_file, "r", encoding="utf-8") as f:
        test_files = f.read().split(" ")

    if len(test_files) == 0 or test_files == [""]:
        print("No tests to filter.")
        return

    if test_files == ["tests"]:
        test_files = [os.path.join("tests", f) for f in os.listdir("tests") if f not in ["__init__.py"] + filters]
    else:
        test_files = [f for f in test_files if f.split(os.path.sep)[1] not in filters]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(" ".join(test_files))


def parse_commit_message(commit_message: str) -> Dict[str, bool]:
    """
    Parses the commit message to detect if a command is there to skip, force all or part of the CI.

    Args:
        commit_message (`str`): The commit message of the current commit.

    Returns:
        `Dict[str, bool]`: A dictionary of strings to bools with keys the following keys: `"skip"`,
        `"test_all_models"` and `"test_all"`.
    """
    if commit_message is None:
        return {"skip": False, "no_filter": False, "test_all": False}

    command_search = re.search(r"\[([^\]]*)\]", commit_message)
    if command_search is not None:
        command = command_search.groups()[0]
        command = command.lower().replace("-", " ").replace("_", " ")
        skip = command in ["ci skip", "skip ci", "circleci skip", "skip circleci"]
        no_filter = set(command.split(" ")) == {"no", "filter"}
        test_all = set(command.split(" ")) == {"test", "all"}
        return {"skip": skip, "no_filter": no_filter, "test_all": test_all}
    else:
        return {"skip": False, "no_filter": False, "test_all": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str, default="test_list.txt", help="Where to store the list of tests to run"
    )
    parser.add_argument(
        "--json_output_file",
        type=str,
        default="test_map.json",
        help="Where to store the tests to run in a dictionary format mapping test categories to test files",
    )
    parser.add_argument(
        "--diff_with_last_commit",
        action="store_true",
        help="To fetch the tests between the current commit and the last commit",
    )
    parser.add_argument(
        "--filter_tests",
        action="store_true",
        help="Will filter the pipeline/repo utils tests outside of the generated list of tests.",
    )
    parser.add_argument(
        "--print_dependencies_of",
        type=str,
        help="Will only print the tree of modules depending on the file passed.",
        default=None,
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        help="The commit message (which could contain a command to force all tests or skip the CI).",
        default=None,
    )
    args = parser.parse_args()
    if args.print_dependencies_of is not None:
        print_tree_deps_of(args.print_dependencies_of)
    else:
        repo = Repo(PATH_TO_REPO)
        commit_message = repo.head.commit.message
        commit_flags = parse_commit_message(commit_message)
        if commit_flags["skip"]:
            print("Force-skipping the CI")
            quit()
        if commit_flags["no_filter"]:
            print("Running all tests fetched without filtering.")
        if commit_flags["test_all"]:
            print("Force-launching all tests")

        diff_with_last_commit = args.diff_with_last_commit
        if not diff_with_last_commit and not repo.head.is_detached and repo.head.ref == repo.refs.main:
            print("main branch detected, fetching tests against last commit.")
            diff_with_last_commit = True

        if not commit_flags["test_all"]:
            try:
                infer_tests_to_run(
                    args.output_file,
                    diff_with_last_commit=diff_with_last_commit,
                    json_output_file=args.json_output_file,
                )
                filter_tests(args.output_file, ["repo_utils"])
                update_test_map_with_core_pipelines(json_output_file=args.json_output_file)

            except Exception as e:
                print(f"\nError when trying to grab the relevant tests: {e}\n\nRunning all tests.")
                commit_flags["test_all"] = True

        if commit_flags["test_all"]:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write("tests")
            example_file = Path(args.output_file).parent / "examples_test_list.txt"
            with open(example_file, "w", encoding="utf-8") as f:
                f.write("all")

            test_files_to_run = get_all_tests()
            create_json_map(test_files_to_run, args.json_output_file)
            update_test_map_with_core_pipelines(json_output_file=args.json_output_file)
