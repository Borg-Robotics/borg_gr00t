"""Embodiment tag resolution for BORG robots.

Translates the user-friendly 'borg' tag to the upstream 'new_embodiment' tag
used by Isaac GR00T N1.6.
"""

from gr00t.data.embodiment_tags import EmbodimentTag


def resolve_embodiment_tag(tag: str) -> str:
    """Translate 'borg' to 'new_embodiment', passthrough for anything else."""
    if tag.lower() == "borg":
        return EmbodimentTag.NEW_EMBODIMENT.value
    return tag


def resolve_embodiment_enum(tag: str) -> EmbodimentTag:
    """Translate 'borg' to EmbodimentTag.NEW_EMBODIMENT, otherwise look up the enum."""
    if tag.lower() == "borg":
        return EmbodimentTag.NEW_EMBODIMENT
    return EmbodimentTag(tag)
