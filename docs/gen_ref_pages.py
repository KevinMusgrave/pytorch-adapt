"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

TOP_LEVEL_NAME = "pytorch_adapt"


def remove_pytorch_adapt(x):
    return [z for z in list(x.parts) if z != TOP_LEVEL_NAME]


def main():
    nav = mkdocs_gen_files.Nav()

    for path in sorted(Path("src").rglob("*.py")):
        module_path = path.relative_to("src").with_suffix("")
        parts = list(module_path.parts)
        if parts[-1] in ["__init__", "__main__"]:
            continue

        doc_path = path.relative_to("src").with_suffix(".md")
        full_doc_path = Path("reference", doc_path)
        doc_path = Path(*remove_pytorch_adapt(doc_path))
        full_doc_path = Path(*remove_pytorch_adapt(full_doc_path))

        for_nav = remove_pytorch_adapt(module_path)
        nav[for_nav] = doc_path

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:
            ident = ".".join(parts)
            print("::: " + ident, file=fd)

        mkdocs_gen_files.set_edit_path(full_doc_path, path)

    with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
        nav_file.writelines(nav.build_literate_nav())


main()
