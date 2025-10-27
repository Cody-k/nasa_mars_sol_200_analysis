"""RAD data parser | Parse NASA Radiation Assessment Detector data files"""

from pathlib import Path
from typing import Iterator
import re


class RADDataParser:
    """Parser for NASA RAD (Radiation Assessment Detector) text data files"""

    def __init__(self, data_file: Path):
        self.data_file = Path(data_file)

    def parse_header(self) -> dict[str, str]:
        """Extract file-level metadata from [FILE] section"""
        header = {}
        with self.data_file.open() as f:
            in_file_section = False
            for line in f:
                line = line.strip()
                if line == "[FILE]":
                    in_file_section = True
                    continue
                if line.startswith("[") and in_file_section:
                    break
                if in_file_section and "=" in line:
                    key, value = line.split("=", 1)
                    header[key] = value.strip('"')
        return header

    def parse_observations(self) -> Iterator[dict[str, str]]:
        """Generate observation records from data file"""
        with self.data_file.open() as f:
            current_obs = {}
            in_obs_section = False

            for line in f:
                line = line.strip()

                if match := re.match(r"\[OBSERVATION: (\d+)\]", line):
                    if current_obs:
                        yield current_obs
                    current_obs = {"obs_id": match.group(1)}
                    in_obs_section = True
                    continue

                if line.startswith("[COUNTERS:") and in_obs_section:
                    in_obs_section = False
                    if current_obs:
                        yield current_obs
                        current_obs = {}
                    continue

                if in_obs_section and "=" in line:
                    key, value = line.split("=", 1)
                    current_obs[key] = value.strip('"')
