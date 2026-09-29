"""H2O/TOVA step-boundary controls on the original inference engine."""
from pathlib import Path
import hashlib
import run_rebuttal_legacy as legacy

_base_source_hashes = legacy.source_hashes


def source_hashes(source):
    hashes = _base_source_hashes(source)
    hashes["__step_entrypoint__"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def main():
    legacy.EXTENSIONS = ("h2o_step", "tova_step")
    legacy.source_hashes = source_hashes
    legacy.main()


if __name__ == "__main__":
    main()
