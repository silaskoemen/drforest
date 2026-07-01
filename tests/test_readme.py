import re
from pathlib import Path
from typing import Any


def test_readme_python_examples_execute():
    readme = Path(__file__).parents[1] / "README.md"
    blocks = re.findall(r"```python\n(.*?)\n```", readme.read_text(), flags=re.DOTALL)
    assert blocks

    namespace: dict[str, Any] = {"__name__": "__readme__"}
    for index, source in enumerate(blocks):
        code = compile(source, f"README.md:python-block-{index}", "exec")
        exec(code, namespace)
