"""Print any saved ChArUco board profile at an exact physical size.

A PNG carries no physical size of its own -- OpenCV writes no resolution metadata -- so
the file this produces is only correct if it is printed at the DPI stated alongside it,
with the printer's "fit to page" scaling turned off. Getting that wrong scales the board
by a few per cent, which is exactly the error that a printed size is supposed to rule
out: the region homography is fitted against the *declared* square size, so a board
printed 4% small produces measurements 4% short with a flawless fit and near-zero
residuals. Nothing downstream can detect it.

So nothing is written without its companion text file, and the text file leads with the
one number that matters -- the size the board must measure on paper once printed.

Kept separate from :mod:`CalibrateAPP.generate_two_boards` rather than folded into it,
because that module's output filenames are asserted by the two-board tests and its boards
belong to a different workflow.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2

from app_config import CONFIG, DATA_ROOT, project_path
import region_calibration

# A4 printable width is about 200 mm, so a board is usually printed on whatever paper
# takes it and cut down. The margin exists so there is something to cut along and so the
# outermost markers have white space around them, which detection wants.
DEFAULT_MARGIN_MM = 25.0
DEFAULT_DPI = 300.0


def board_filename(profile):
    """A stable filename for a profile, safe on every filesystem."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(profile.name).strip().lower()).strip("_")
    return f"{slug or 'board'}.png"


def board_pixels(profile, dpi=DEFAULT_DPI, margin_mm=DEFAULT_MARGIN_MM):
    """Pixel size of the board, of its margin, and of the whole sheet at this DPI."""
    pixels_per_mm = float(dpi) / 25.4
    width_mm, height_mm = profile.footprint_mm()
    board = (int(round(width_mm * pixels_per_mm)),
             int(round(height_mm * pixels_per_mm)))
    margin = int(round(float(margin_mm) * pixels_per_mm))
    return board, margin, (board[0] + 2 * margin, board[1] + 2 * margin)


def board_profiles_path():
    return Path(project_path(CONFIG.get("region_homography", {}).get(
        "board_profiles_file", "Files/board_definitions.json")))


def load_profiles(path=None):
    return region_calibration.load_board_profiles(path or board_profiles_path())


def generate_board(profile, output_dir=None, dpi=DEFAULT_DPI,
                   margin_mm=DEFAULT_MARGIN_MM):
    """Write one profile as a PNG plus the note that says how to print it.

    Returns ``(image_path, note_path)``.
    """
    profile.validate()
    board, margin, sheet = board_pixels(profile, dpi, margin_mm)
    width_mm, height_mm = profile.footprint_mm()
    output_dir = Path(project_path(output_dir or 'Files/calibration_boards'))
    output_dir.mkdir(parents=True, exist_ok=True)

    image = profile.build_board().generateImage(sheet, marginSize=margin, borderBits=1)
    image_path = output_dir / board_filename(profile)
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"Could not write {image_path}")

    note_path = image_path.with_suffix(".txt")
    note_path.write_text(
        "\n".join([
            f"Board:      {profile.name}",
            f"Definition: {profile.describe()}",
            "",
            f"PRINT AT EXACTLY {width_mm:.1f} x {height_mm:.1f} mm",
            "  (that is the board itself, not the sheet)",
            "",
            f"Printer setting: {dpi:.0f} DPI",
            "  Turn OFF 'fit to page', 'scale to fit' or any automatic scaling.",
            "  On A4 at 100%, the board may not fit; use larger paper or a print shop.",
            "",
            f"The sheet including its {margin_mm:.0f} mm margin measures "
            f"{sheet[0]} x {sheet[1]} px "
            f"({sheet[0] / (dpi / 25.4):.1f} x {sheet[1] / (dpi / 25.4):.1f} mm).",
            "",
            "After printing, measure one square with a ruler. If it is not "
            f"{profile.square_length_mm:g} mm, the print was scaled and the board cannot "
            "be used -- a scaled board fits perfectly and measures wrongly.",
            "",
        ]),
        encoding="utf-8",
    )
    return image_path, note_path


def generate_all(profiles=None, output_dir=None, dpi=DEFAULT_DPI,
                 margin_mm=DEFAULT_MARGIN_MM):
    """Write every usable stored profile. Returns the paths that were written."""
    written = []
    for profile in (profiles if profiles is not None else load_profiles()):
        try:
            written.append(generate_board(profile, output_dir, dpi, margin_mm))
        except (ValueError, RuntimeError, cv2.error) as error:
            # One unusable profile must not stop the others from printing; the operator
            # sees which one failed rather than an empty output directory.
            print(f"Skipping '{profile.name}': {error}", file=sys.stderr)
    return written


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*",
                        help="profile names to print; all saved profiles by default")
    parser.add_argument("--dpi", type=float, default=DEFAULT_DPI)
    parser.add_argument("--margin-mm", type=float, default=DEFAULT_MARGIN_MM)
    parser.add_argument("--output-dir", default=None)
    arguments = parser.parse_args(argv)

    profiles = load_profiles()
    if arguments.names:
        wanted = {name.strip().lower() for name in arguments.names}
        chosen = [p for p in profiles if p.name.strip().lower() in wanted]
        missing = wanted - {p.name.strip().lower() for p in chosen}
        if missing:
            print(f"No such board profile: {', '.join(sorted(missing))}", file=sys.stderr)
            print("Known profiles: " + ", ".join(p.name for p in profiles),
                  file=sys.stderr)
            return 2
    else:
        chosen = profiles

    written = generate_all(chosen, arguments.output_dir, arguments.dpi,
                           arguments.margin_mm)
    for image_path, note_path in written:
        print(image_path)
        print(note_path)
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
