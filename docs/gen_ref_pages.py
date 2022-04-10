"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files
from griffe.collections import ModulesCollection
from griffe.dataclasses import Alias, Module
from griffe.loader import GriffeLoader

TOP_LEVEL_NAME = "pytorch_adapt"
FOLDER = "code"


def remove_pytorch_adapt(x):
    return [z for z in list(x.parts) if z != TOP_LEVEL_NAME]


def get_init_entries(module_instance, init_entries, prefix=""):
    for k, v in module_instance.members.items():
        if isinstance(v, Module) and v.is_init_module:
            init_key = f"{prefix}.{k}" if prefix else k
            init_entries[init_key] = sorted(
                [
                    (name, member.target_path)
                    for name, member in v.members.items()
                    if isinstance(member, Alias)
                ]
            )
            get_init_entries(v, init_entries, prefix=init_key)
    return init_entries


def get_init_doc_contents(module_name, members, collection):
    output = ""
    for name, target_path in members:
        if collection[target_path].has_docstring:
            output += f"- [{name}][{target_path}]\n"
        else:
            output += f"- {name}\n"
    if len(output) > 0:
        example_name, _ = members[0]
        prepend = (
            f"The following can be imported like this (using ```{example_name}``` as an example):\n\n"
            f"```from {TOP_LEVEL_NAME}.{module_name} import {example_name}```\n\n"
            "## Direct module members"
        )
        output = f"{prepend}\n\n{output}"
    return output


def set_init_pages(module_instance, collection, nav):
    init_entries = get_init_entries(module_instance, {}, prefix="")
    for k, v in init_entries.items():
        k_split = k.split(".")
        doc_path = Path(*k_split, "index").with_suffix(".md")
        full_doc_path = Path(FOLDER, doc_path)
        nav[k_split] = doc_path
        to_write = get_init_doc_contents(k, v, collection)
        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            fd.write(to_write)


def set_non_init_pages(collection, nav):
    for path in sorted(Path("src").rglob("*.py")):
        module_path = path.relative_to("src").with_suffix("")
        parts = tuple(module_path.parts)
        if (parts[-1] in ["__init__", "__main__"]) or (
            not collection[module_path.parts].has_docstrings
        ):
            continue

        doc_path = path.relative_to("src").with_suffix(".md")
        doc_path = Path(*remove_pytorch_adapt(doc_path))
        full_doc_path = Path(FOLDER, doc_path)

        for_nav = remove_pytorch_adapt(module_path)
        nav[for_nav] = doc_path

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            fd.write(f"::: {ident}")

        mkdocs_gen_files.set_edit_path(full_doc_path, path)


def main():
    collection = ModulesCollection()
    loader = GriffeLoader(modules_collection=collection)
    module_instance = loader.load_module(Path("src", TOP_LEVEL_NAME))
    nav = mkdocs_gen_files.Nav()
    set_init_pages(module_instance, collection, nav)
    set_non_init_pages(collection, nav)
    with mkdocs_gen_files.open(f"{FOLDER}/SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


main()
