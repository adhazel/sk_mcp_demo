
import inspect, __main__
from pathlib import Path

def get_caller():
    # 1) inspect stack for first non-this‐module frame
    for frame_info in inspect.stack()[1:]:
        fname = Path(frame_info.filename)
        # skip internal frames (optional)
        if fname.stem != Path(__file__).stem:
            return fname.resolve()

    # 2) fallback to entrypoint script
    entry = getattr(__main__, "__file__", None)
    return Path(entry).resolve() if entry else None

if __name__ == "__main__":
    print("Caller:", get_caller())
