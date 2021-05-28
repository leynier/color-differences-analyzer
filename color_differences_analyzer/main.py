from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, List, Tuple, Union, cast

import numpy as np
import pandas as pd
import typer
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from skimage import color, io

app = typer.Typer()


class OutputType(str, Enum):
    none = "none"
    excel = "excel"


file_or_dir_argument = typer.Argument(
    ...,
    exists=True,
    file_okay=True,
    dir_okay=True,
    writable=False,
    readable=True,
    resolve_path=True,
)

output_option = typer.Option(OutputType.none, case_sensitive=False)


@app.command(name="dist2rgb")
def dist2rgb_command(
    file_or_dir: Path = file_or_dir_argument,
    output: OutputType = output_option,
):
    dist2_command(
        file_or_dir,
        output,
        dist2rgb,
        [
            "Red Distances",
            "Green Distances",
            "Blue Distances",
        ],
    )


@app.command(name="dist2red")
def dist2red_command(
    file_or_dir: Path = file_or_dir_argument,
    output: OutputType = output_option,
):
    dist2_command(file_or_dir, output, dist2red, ["Distances"])


@app.command(name="dist2green")
def dist2green_command(
    file_or_dir: Path = file_or_dir_argument,
    output: OutputType = output_option,
):
    dist2_command(file_or_dir, output, dist2green, ["Distances"])


@app.command(name="dist2blue")
def dist2blue_command(
    file_or_dir: Path = file_or_dir_argument,
    output: OutputType = output_option,
):
    dist2_command(file_or_dir, output, dist2blue, ["Distances"])


def dist2_command(
    file_or_dir: Path,
    output: OutputType,
    func: Callable[[str], Tuple[Union[float, List[float]], timedelta]],
    labels: List[str],
):
    if file_or_dir.is_file():
        distance, time = func(str(file_or_dir))
        typer.echo(f"{file_or_dir.name}: {distance}. Time: {time}")
        if output == OutputType.excel:
            to_excel(
                [
                    (
                        [cast(float, distance)]
                        if isinstance(distance, float)
                        else cast(List[float], distance),
                        time,
                    )
                ],
                [file_or_dir],
                labels,
            )
    else:
        filenames = [item for item in file_or_dir.iterdir()]
        total = len(filenames)
        results: List[Tuple[Union[float, List[float]], timedelta]] = []
        for i, filename in enumerate(filenames):
            results.append(func(str(filename)))
            distance, time = results[-1]
            typer.echo(f"({i+1}/{total}) {filename.name}: {distance}. Time: {time}")
        if output == OutputType.excel:
            to_excel(
                [
                    (
                        [cast(float, item[0])]
                        if isinstance(item[0], float)
                        else cast(List[float], item[0]),
                        item[1],
                    )
                    for item in results
                ],
                filenames,
                labels,
            )


def dist2rgb(filename: str) -> Tuple[List[float], timedelta]:
    red = sRGBColor(1.0, 0.0, 0.0)
    green = sRGBColor(0.0, 1.0, 0.0)
    blue = sRGBColor(0.0, 0.0, 1.0)
    return dist2colors(filename, [red, green, blue])


def dist2red(filename: str) -> Tuple[float, timedelta]:
    color = sRGBColor(1.0, 0.0, 0.0)
    distances, time = dist2colors(filename, [color])
    return distances[0], time


def dist2green(filename: str) -> Tuple[float, timedelta]:
    color = sRGBColor(0.0, 1.0, 0.0)
    distances, time = dist2colors(filename, [color])
    return distances[0], time


def dist2blue(filename: str) -> Tuple[float, timedelta]:
    color = sRGBColor(0.0, 0.0, 1.0)
    distances, time = dist2colors(filename, [color])
    return distances[0], time


def dist2colors(
    filename: str,
    ref_colors: List[Union[sRGBColor, LabColor]],
) -> Tuple[List[float], timedelta]:
    start_datetime = datetime.now()
    image_argb: np.ndarray = io.imread(filename)
    image_rgb: np.ndarray = image_argb[:, :, :3]
    image_lab: np.ndarray = color.rgb2lab(image_rgb)
    image_lab_flatted = image_lab.reshape(
        (image_lab.shape[0] * image_lab.shape[1], image_lab.shape[2])
    )
    image_lab_flatted_filter = image_lab_flatted != [0.0, 0.0, 0.0]
    image_lab_flatted_filter = np.array([np.any(x) for x in image_lab_flatted_filter])
    image_lab_flatted_filtered: np.ndarray = image_lab_flatted[image_lab_flatted_filter]
    image_lab_mean: np.ndarray = np.mean(image_lab_flatted_filtered, 0)
    image_lab_mean_color = LabColor(*image_lab_mean)
    distances: List[float] = []
    for ref_color in ref_colors:
        ref_lab_color = (
            cast(LabColor, convert_color(ref_color, LabColor))
            if isinstance(ref_color, sRGBColor)
            else cast(LabColor, ref_color)
        )
        distance = delta_e_cie2000(image_lab_mean_color, ref_lab_color)
        distances.append(distance)
    return distances, datetime.now() - start_datetime


def to_excel(
    items: List[Tuple[List[float], timedelta]],
    filenames: List[Path],
    labels: List[str],
    excel_name="result.xlsx",
):
    _index = [item.name for item in filenames]
    _labels = labels + ["Times"]
    _series = [list(item) for item in zip(*[item for item, _ in items])] + [
        [item.total_seconds() for _, item in items]
    ]
    df = pd.DataFrame(
        {
            label: pd.Series(serie, index=_index)
            for label, serie in zip(_labels, _series)
        }
    )
    df.to_excel(excel_name)
