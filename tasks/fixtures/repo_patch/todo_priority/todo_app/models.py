from dataclasses import dataclass


@dataclass
class Todo:
    title: str
    done: bool = False
    tags: tuple[str, ...] = ()
