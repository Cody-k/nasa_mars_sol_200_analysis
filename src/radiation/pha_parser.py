"""PHA parser | Extract pulse height analysis events from RAD data"""

from pathlib import Path
import re
import polars as pl


class PHAParser:
    """Parse PHA (Pulse Height Analysis) data from RAD instrument files"""

    def __init__(self):
        self.pha_pattern = re.compile(r"^\[PHA: (\d+)\]$")

    def parse_pha_events(self, rad_file: Path, debug: bool = False) -> pl.DataFrame:
        """
        Extract all PHA events from RAD file

        Returns DataFrame with:
        - event_id: Sequential event number
        - sclk: Spacecraft clock time
        - utc: UTC timestamp string
        - priority: Event priority (0-1)
        - hw_priority: Hardware priority
        - l2_mask: Logic level 2 mask (binary string)
        - slow_token_mask: Slow token mask (binary string)
        - rad_00 to rad_35: Raw detector channel values
        - corr_00 to corr_35: Corrected detector channel values
        """
        lines = rad_file.read_text().splitlines()

        events = []
        current_event_id = None
        in_data_section = False
        data_line_count = 0

        for i, line in enumerate(lines):
            pha_match = self.pha_pattern.match(line.strip())

            if pha_match:
                if debug and current_event_id is not None:
                    print(f"Event {current_event_id}: {data_line_count} data lines parsed")
                current_event_id = int(pha_match.group(1))
                in_data_section = False
                data_line_count = 0
                if debug:
                    print(f"Line {i}: Found PHA event {current_event_id}")
                continue

            if current_event_id is not None:
                if "sclk" in line and "utc" in line:
                    if debug:
                        print(f"Line {i}: Skipping header")
                    continue

                if "---" in line:
                    in_data_section = True
                    if debug:
                        print(f"Line {i}: Entering data section")
                    continue

                if in_data_section:
                    if line.strip() and not line.strip().startswith("["):
                        event_data = self._parse_event_line(line, current_event_id)
                        if event_data:
                            events.append(event_data)
                            data_line_count += 1
                            if debug and data_line_count <= 2:
                                print(f"Line {i}: Parsed data line #{data_line_count}")

        if debug:
            print(f"Total events parsed: {len(events)}")

        if not events:
            return pl.DataFrame()

        df = pl.DataFrame(events)

        df = df.with_columns([
            pl.col("sclk").cast(pl.Float64),
            pl.col("priority").cast(pl.Int32),
            pl.col("hw_priority").cast(pl.Int32),
        ])

        return df

    def _parse_event_line(self, line: str, event_id: int) -> dict:
        """Parse a single PHA event data line"""
        parts = line.split()

        if len(parts) < 78:
            return None

        try:
            event = {
                "event_id": event_id,
                "sclk": float(parts[0]),
                "utc": parts[1].strip('"'),
                "priority": int(parts[2]),
                "hw_priority": int(parts[3]),
                "l2_mask": parts[4].strip('"'),
                "slow_token_mask": parts[5].strip('"'),
            }

            for i in range(36):
                rad_idx = 6 + (i * 2)
                corr_idx = 6 + (i * 2) + 1

                event[f"rad_{i:02d}"] = float(parts[rad_idx])
                event[f"corr_{i:02d}"] = float(parts[corr_idx])

            return event
        except (ValueError, IndexError):
            return None

    def get_valid_events(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter for valid PHA events

        Removes events where all detector channels are -1 or -2 (no signal)
        """
        channel_cols = [f"corr_{i:02d}" for i in range(36)]

        df = df.with_columns(
            pl.concat_str([pl.col(c) for c in channel_cols]).alias("channel_str")
        )

        valid_mask = ~pl.col("channel_str").str.contains("^(-1\\.0|-2\\.0)+$")

        return df.filter(valid_mask).drop("channel_str")

    def calculate_total_energy(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate total energy deposited across all detectors

        Sums corrected values from all channels (excluding negative values)
        """
        channel_cols = [f"corr_{i:02d}" for i in range(36)]

        for col in channel_cols:
            df = df.with_columns(
                pl.when(pl.col(col) < 0)
                .then(0.0)
                .otherwise(pl.col(col))
                .alias(col + "_pos")
            )

        pos_cols = [c + "_pos" for c in channel_cols]

        df = df.with_columns(
            pl.sum_horizontal(pos_cols).alias("total_energy")
        )

        df = df.drop(pos_cols)

        return df

    def calculate_detector_counts(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Count number of detectors triggered per event

        Counts channels with positive values (signal detected)
        """
        channel_cols = [f"corr_{i:02d}" for i in range(36)]

        for col in channel_cols:
            df = df.with_columns(
                (pl.col(col) > 0).cast(pl.Int32).alias(col + "_triggered")
            )

        trigger_cols = [c + "_triggered" for c in channel_cols]

        df = df.with_columns(
            pl.sum_horizontal(trigger_cols).alias("n_detectors_triggered")
        )

        df = df.drop(trigger_cols)

        return df
