#!/usr/bin/env python3

"""
Locker Problem Solver

A comprehensive and educational-quality solution to the classic Locker Problem. This script provides multiple
approaches, including a highly efficient mathematical solution and a step-by-step simulation, complete with performance
benchmarking and an interactive explorer.

The Locker Problem:
    A building has a row of lockers, all initially closed. In the first pass, you open every locker. In the second pass,
    you toggle every second locker (closing it if open, opening it if closed). In the third pass, you toggle every third
    locker, and so on, until you have made a pass for every locker.

Mathematical Insight:
    A locker's final state (open or closed) depends on how many times it was toggled. A locker is toggled once for each
    of its divisors.

    If a locker number has an EVEN number of divisors, it will be toggled an even number of times, returning it to its
    initial CLOSED state.

    If a locker number has an ODD number of divisors, it will be toggled an odd number of times, leaving it in the OPEN
    state.

    Only perfect squares have an odd number of divisors. This is because divisors of non-perfect-square numbers always
    come in pairs (e.g., for 12: 1&12, 2&6, 3&4). For a perfect square like 36, the pairs are 1&36, 2&18, 3&12, and 4&9,
    but the square root, 6, is paired with itself, resulting in an odd count.

Complexity Analysis:
    - Mathematical Solution: O(√n) Time, O(√n) Space
      (To find all perfect squares up to n)
    - Simulation Solution: O(n log(n)) Time, O(n) Space
      (The simulation solution involves a sum of n/k for k=1 to n, which is a harmonic series approximating to n log(n))

Structure of the Code:
    - Constants and Helpers: Basic configuration, type definitions, and helper functions for validating and UI printing.
    - MathematicalSolution: The efficient O(√n) solution.
    - SimulationSolution: The O(n log(n)) step-by-step simulation solution.
    - InteractiveExplorer: A command-line interface for exploring the problem.
    - Benchmark Tools: `BenchmarkResult` and `PerformanceBenchmark` for performance analysis.
    - Main Execution: `argparse` setup and main script logic.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, NotRequired, TypedDict, TypeVar

import argparse
import functools
import json
import math
import statistics
import sys
import time

APP_NAME: str = "Locker Problem Solver"
__version__: str = "1.0.0"

UI_WIDTH: int = 60
EMOJI_LOCK: str = "🔐"

T = TypeVar("T")


class BenchmarkResultDict(TypedDict):
    """A dictionary representation of a benchmark result, suitable for JSON serialization."""

    method_name: str
    count: int
    repeats: int
    avg_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    durations_ms: list[float]
    open_lockers: NotRequired[list[int]]


def _format_int(n: int) -> str:
    """
    Return a human-readable string for an integer, with thousands separators.

    Args:
        n: The integer to format.

    Returns:
        A formatted string with thousands separators.
    """

    return f"{n:,}"


def _print_banner() -> None:
    """Print the application banner."""

    print(EMOJI_LOCK * (UI_WIDTH // 2))
    print(f"{APP_NAME} v{__version__}".center(UI_WIDTH))
    print(EMOJI_LOCK * (UI_WIDTH // 2))


def _print_divider(width: int = UI_WIDTH, fill: str = "-") -> None:
    """
    Print a divider line with a custom fill character.

    Args:
        width: The width of the divider line (default: `UI_WIDTH`).
        fill: The character used to draw the line (default: `-`).
    """

    print(fill * width)


def _print_header(title: str, width: int = UI_WIDTH, fill: str = "=") -> None:
    """
    Print a distinct header with a title and custom fill character.

    Args:
        title: The header title to print.
        width: The width of the header line (default: `UI_WIDTH`).
        fill: The character used to draw the line (default: `=`).
    """

    line: str = fill * width
    print(f"\n{line}")
    print(f"{title}".center(width))
    print(line)


def _print_section_title(title: str) -> None:
    """
    Print a formatted section title with a divider.

    Args:
        title: The section title to print.
    """

    print(f"\n{title}")
    _print_divider(fill="=")


def _validate_positive_integer(value: Any, name: str) -> None:
    """
    Validate that a value is a positive integer.

    Args:
        value: The value to validate.
        name: The name of the parameter for clear error messages.

    Raises:
        TypeError: If `value` is not an integer.
        ValueError: If `value` is not positive.
    """

    if type(value) is not int:
        raise TypeError(f"`{name}` must be an integer, but got `{type(value).__name__}`.")
    elif value <= 0:
        raise ValueError(f"`{name}` must be positive, but got `{value}`.")


def _validate_locker_range(locker_num: int, total_lockers: int) -> None:
    """
    Validate that a locker number is within the valid (1-indexed) range.

    Args:
        locker_num: The locker number to validate.
        total_lockers: The total number of lockers available.

    Raises:
        TypeError: If `locker_num` is not an integer.
        ValueError: If `locker_num` is out of the valid range [1, total_lockers].
    """

    _validate_positive_integer(locker_num, "locker_num")

    if not (1 <= locker_num <= total_lockers):
        raise ValueError(f"Locker number `{locker_num}` is out of range. Must be between 1 and `{total_lockers}`.")


class MathematicalSolution:
    """
    Mathematical solution using the perfect squares insight.

    This class provides an efficient O(√n) solution based on the mathematical insight that only lockers numbered as
    perfect squares will remain open.

    The reasoning: A locker is toggled once for each of its divisors. Most numbers have divisors that pair up (e.g., 12
    has pairs: 1×12, 2×6, 3×4), resulting in an even number of divisors. Perfect squares have one unpaired divisor
    (the square root), resulting in an odd number of divisors and thus remaining open.
    """

    @staticmethod
    def solve(num_lockers: int) -> list[int]:
        """
        Calculate the open lockers using the perfect square insight.

        Args:
            num_lockers: The total number of lockers.

        Returns:
            A sorted list of open locker numbers (all perfect squares up to `num_lockers`).

        Raises:
            TypeError: If `num_lockers` is not an integer.
            ValueError: If `num_lockers` is not positive.
        """

        _validate_positive_integer(num_lockers, "num_lockers")

        # Find all perfect squares up to `num_lockers` by finding the largest integer square root and squaring all
        # integers from 1 up to that root.
        root: int = math.isqrt(num_lockers)
        return [i * i for i in range(1, root + 1)]

    @staticmethod
    def count_open_lockers(num_lockers: int) -> int:
        """
        Count the number of open lockers using the perfect square insight.

        This is the most efficient way to determine the count, as it directly calculates how many perfect squares exist
        up to `num_lockers` without generating them.

        Args:
            num_lockers: The total number of lockers.

        Returns:
            The number of open lockers.

        Raises:
            TypeError: If `num_lockers` is not an integer.
            ValueError: If `num_lockers` is not positive.
        """

        _validate_positive_integer(num_lockers, "num_lockers")

        # The number of perfect squares up to n is simply floor(√n).
        return math.isqrt(num_lockers)

    @staticmethod
    def is_locker_open(locker_num: int) -> bool:
        """
        Check if a specific locker is open using the perfect square insight.

        Args:
            locker_num: The locker number to check (1-indexed).

        Returns:
            True if the locker is open (i.e., `locker_num` is a perfect square), False otherwise.

        Raises:
            TypeError: If `locker_num` is not an integer.
            ValueError: If `locker_num` is not positive.
        """

        _validate_positive_integer(locker_num, "locker_num")

        # A number is a perfect square if its integer square root, when squared, equals the original number.
        root: int = math.isqrt(locker_num)
        return root * root == locker_num

    @staticmethod
    def get_perfect_square_info(locker_num: int) -> tuple[bool, int | None]:
        """
        Provide detailed information about whether a locker number is a perfect square.

        Args:
            locker_num: The locker number to check.

        Returns:
            A tuple `(is_perfect_square, square_root)` where:
                - `is_perfect_square`: True if `locker_num` is a perfect square, False otherwise.
                - `square_root`: The integer square root if it is a perfect square, None otherwise.

        Raises:
            TypeError: If `locker_num` is not an integer.
            ValueError: If `locker_num` is not positive.
        """

        _validate_positive_integer(locker_num, "locker_num")

        root: int = math.isqrt(locker_num)
        if root * root == locker_num:
            return True, root

        return False, None

    @staticmethod
    def squares_in_range(start: int, end: int) -> list[int]:
        """
        Find all perfect squares within an inclusive range [start, end].

        Args:
            start: The start of the range (inclusive).
            end: The end of the range (inclusive).

        Returns:
            A sorted list of perfect squares within the range.

        Raises:
            TypeError: If `start` or `end` are not integers.
            ValueError: If `start` or `end` are not positive or `start` is greater than `end`.
        """

        _validate_positive_integer(start, "start")
        _validate_positive_integer(end, "end")

        if start > end:
            raise ValueError("`start` must be less than or equal to `end`.")

        # Find the first integer whose square is greater than or equal to `start`.
        # `math.isqrt(start - 1) + 1` is an efficient, integer-only way to compute ceil(sqrt(start)).
        first_root: int = math.isqrt(start - 1) + 1
        # The last integer whose square is less than or equal to `end` is just floor(√(end)).
        last_root: int = math.isqrt(end)

        return [i * i for i in range(first_root, last_root + 1)]


class InteractiveExplorer:
    """Provides interactive exploration of the Locker Problem."""

    # Command strings
    _CMD_QUIT: tuple[str, ...] = ("q", "quit", "exit")
    _CMD_HELP: str = "help"
    _CMD_RANGE: str = "range"
    _CMD_COUNT: str = "count"

    # This limit prevents excessive console output for very large ranges.
    MAX_RANGE_SPAN: int = 100_000

    @staticmethod
    def _parse_int(token: str) -> int:
        """
        Parse an integer from a string, allowing separators like commas or underscores.

        Args:
            token: The string token to parse.

        Returns:
            The parsed integer.

        Raises:
            ValueError: If the token cannot be converted to an integer.

        Examples:
            _parse_int("1,000") -> 1000
            _parse_int("2_500") -> 2500
        """

        cleaned_token: str = token.replace(",", "").replace("_", "").strip()
        return int(cleaned_token)

    @staticmethod
    def _handle_single_locker_command(input_str: str) -> None:
        """
        Handle a query to check if a single, specific locker is open.

        Args:
            input_str: User input string, which should be parsable as an integer.
        """

        locker_num: int = InteractiveExplorer._parse_int(input_str)
        is_perfect, root = MathematicalSolution.get_perfect_square_info(locker_num)

        status: str = "🟢 OPEN" if is_perfect else "🔴 CLOSED"
        print(f"Locker {_format_int(locker_num)}: {status}")

        if is_perfect:
            print(f"  └─ It's A Perfect Square: {_format_int(root)}² = {_format_int(locker_num)}")
        else:
            sqrt_val: float = math.sqrt(locker_num)
            print(f"  └─ It's Not A Perfect Square, √{_format_int(locker_num)} ≈ {sqrt_val:.3f}")

    @staticmethod
    def _handle_count_command(input_str: str) -> None:
        """
        Handle a `count` command to count open lockers up to a specified number.

        Args:
            input_str: User input string, e.g., "count 1000"

        Raises:
            ValueError: If the input format is incorrect or `number` is invalid.
        """

        parts: list[str] = input_str.split()
        if len(parts) != 2:
            raise ValueError("Invalid format. Use `count <number>`.")

        num_lockers: int = InteractiveExplorer._parse_int(parts[1])
        count: int = MathematicalSolution.count_open_lockers(num_lockers)
        percentage: float = (count / num_lockers) * 100 if num_lockers > 0 else 0.0

        print(f"Open Lockers Up To {_format_int(num_lockers)}: {_format_int(count)} ({percentage:.2f}%)")

    @staticmethod
    def _handle_range_command(input_str: str) -> None:
        """
        Handle a `range` command to check open lockers in a specified range.

        Args:
            input_str: User input string, e.g., "range 1 100".

        Raises:
            ValueError: If the input format is incorrect or the range is invalid.
        """

        parts: list[str] = input_str.split()
        if len(parts) != 3:
            raise ValueError("Invalid format. Use `range <start> <end>`.")

        start, end = InteractiveExplorer._parse_int(parts[1]), InteractiveExplorer._parse_int(parts[2])
        if start > end:
            start, end = end, start  # Normalize order

        span: int = end - start + 1
        if span > InteractiveExplorer.MAX_RANGE_SPAN:
            raise ValueError(
                f"Range is too large. The maximum span is {_format_int(InteractiveExplorer.MAX_RANGE_SPAN)} lockers."
            )

        open_lockers: list[int] = MathematicalSolution.squares_in_range(start, end)

        print(f"Open Lockers In Range {_format_int(start)}-{_format_int(end)}: ", end="")
        if open_lockers:
            print(", ".join(map(_format_int, open_lockers)))
        else:
            print("None.")

        print(f"Total Found: {_format_int(len(open_lockers))} open lockers in a span of {_format_int(span)}.")

    @staticmethod
    def _show_help() -> None:
        """Show detailed help information."""

        _print_header("Help - Interactive Commands")
        print("Single Locker Check:")
        print("  Syntax: <number>")
        print("  - Example: 42          → Checks if locker 42 is open.")
        print("  - Example: 1,000       → Checks if locker 1,000 is open.")
        print()
        print("Range Checking:")
        print("  Syntax: range <start> <end>")
        print("  - Example: range 1 100 → Shows all open lockers from 1 to 100.")
        print()
        print("Counting Open Lockers:")
        print("  Syntax: count <number>")
        print("  - Example: count 50    → Counts how many lockers are open up to 50.")
        print()
        print("Other Commands:")
        print("  - help:                → Show this help message.")
        print("  - quit:                → Exit the interactive explorer.")
        _print_divider(fill="=")

    @staticmethod
    def run() -> None:
        """Run the interactive exploration mode."""

        _print_header("Interactive Locker Explorer")
        print("Enter a command to explore the Locker Problem.")
        print("  - `<number>`: Check if a specific locker is open (e.g., `100`).")
        print("  - `range <start> <end>`: Find open lockers in a range (e.g., `range 1 100`).")
        print("  - `count <number>`: Count open lockers up to a number (e.g., `count 1000`).")
        print("  - `help`: Show detailed help.")
        print("  - `quit` or `q`: Exit.")

        while True:
            try:
                raw: str = input("\n> ")
                if not (user_input := raw.strip()):
                    continue

                cmd_lc: str = user_input.lower()
                if cmd_lc in InteractiveExplorer._CMD_QUIT:
                    print("Goodbye! 👋")
                    break
                elif cmd_lc == InteractiveExplorer._CMD_HELP:
                    InteractiveExplorer._show_help()
                elif cmd_lc.startswith(InteractiveExplorer._CMD_RANGE):
                    InteractiveExplorer._handle_range_command(user_input)
                elif cmd_lc.startswith(InteractiveExplorer._CMD_COUNT):
                    InteractiveExplorer._handle_count_command(user_input)
                else:
                    # Fallback to a single locker check. If it fails to parse, assume it was an unknown command.
                    try:
                        InteractiveExplorer._handle_single_locker_command(user_input)
                    except ValueError:
                        # Re-raise with a more user-friendly message.
                        raise ValueError(f"Unknown command or invalid number `{user_input}`. Type `help` for guidance.")
            except (TypeError, ValueError) as e:
                print(f"❌ Error: {e}")
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"❌ Unexpected Error: {e}")


class SimulationSolution:
    """
    Simulates the Locker Problem by explicitly performing all toggle operations.

    Attributes:
        total_lockers: The total number of lockers in the simulation.

    Example:
        >>> sim: SimulationSolution = SimulationSolution(100)
        >>> sim.execute_simulation()
        >>> open_lockers: list[int] = sim.get_open_lockers()
        >>> print(f"Open Lockers: {open_lockers}")
    """

    # __slots__ tells Python not to use a __dict__ for each instance, saving memory. This is beneficial in
    # memory-constrained environments or when creating a very large number of instances.
    __slots__: tuple[str, str] = ("_total_lockers", "_lockers")

    def __init__(self, total_lockers: int = 1_000) -> None:
        """
        Initialize the locker simulation.

        Args:
            total_lockers: The total number of lockers to include in the simulation (default: 1,000).

        Raises:
            TypeError: If `total_lockers` is not an integer.
            ValueError: If `total_lockers` is not positive.
        """

        _validate_positive_integer(total_lockers, "total_lockers")

        self._total_lockers: int = total_lockers
        self._lockers: list[bool] = [False] * total_lockers  # All lockers start closed.

    def __repr__(self) -> str:
        """
        Return a developer-friendly representation of the simulation instance.

        Returns:
            A string representation of the instance.
        """

        return f"{self.__class__.__name__}(total_lockers={self._total_lockers})"

    @property
    def total_lockers(self) -> int:
        """
        Get the total number of lockers in the simulation.

        Returns:
            The total number of lockers.
        """

        return self._total_lockers

    def execute_simulation(self, show_progress: bool = False) -> None:
        """
        Execute the full locker toggling simulation over all passes.

        In pass `n`, every `n`th locker is toggled. This method performs all passes from 1 to `total_lockers`.

        Optimization Note:
            To maximize performance, this method operates directly on the internal `_lockers` list using pre-validated,
            bounded indices. This avoids the overhead of calling public, validating methods inside the hot loop.

        Args:
            show_progress: If True, display progress updates for large simulations.
        """

        if show_progress and self._total_lockers > 1_000:
            print(f"Starting simulation with {_format_int(self._total_lockers)} lockers...")

        progress_interval: int = max(1, self._total_lockers // 10)
        lockers: list[bool] = self._lockers
        n: int = self._total_lockers

        # For each pass (from 1 to n)...
        for pass_num in range(1, n + 1):
            # ...toggle every `pass_num`-th locker.
            # The loop starts at `pass_num - 1` (for 0-based indexing) and steps by `pass_num`.
            for i in range(pass_num - 1, n, pass_num):
                lockers[i] = not lockers[i]

            # Show progress for large simulations.
            if show_progress and n > 1_000 and pass_num % progress_interval == 0:
                progress: float = (pass_num / n) * 100
                print(f"Simulation Progress: {progress:.1f}%")

    def get_locker_states(self) -> list[bool]:
        """
        Get the current state of all lockers.

        Returns:
            A copy of the list of boolean values representing locker states (False=closed, True=open).
        """

        return self._lockers.copy()

    def get_open_lockers(self) -> list[int]:
        """
        Get all open locker numbers.

        Returns:
            A sorted list of 1-indexed open locker numbers.
        """

        return [i + 1 for i, is_open in enumerate(self._lockers) if is_open]

    def is_locker_open(self, locker_num: int) -> bool:
        """
        Check if a specific locker is open after the simulation has completed.

        Args:
            locker_num: The 1-indexed locker number to check.

        Returns:
            True if the locker is open, False if closed.

        Raises:
            TypeError: If `locker_num` is not an integer.
            ValueError: If `locker_num` is out of the valid range.
        """

        _validate_locker_range(locker_num, self._total_lockers)

        return self._lockers[locker_num - 1]

    def reset(self) -> None:
        """Reset all lockers to the closed state."""

        self._lockers = [False] * self._total_lockers


@dataclass(frozen=True)
class BenchmarkResult:
    """An immutable data class for storing and analyzing benchmark results."""

    method_name: str
    open_lockers: list[int]
    durations_ms: list[float]

    def __repr__(self) -> str:
        """
        Return a developer-friendly representation of the benchmark result.

        Returns:
            A string representation of the instance.
        """

        return (
            f"{self.__class__.__name__}("
            f"method_name='{self.method_name}', "
            f"count={self.count}, "
            f"repeats={len(self.durations_ms)})"
        )

    @property
    def count(self) -> int:
        """Get the count of open lockers."""

        return len(self.open_lockers)

    @functools.cached_property
    def avg_ms(self) -> float:
        """
        Calculate the average execution time in milliseconds.

        Returns:
            The average execution time in milliseconds.
        """

        return statistics.mean(self.durations_ms) if self.durations_ms else 0.0

    @functools.cached_property
    def std_ms(self) -> float:
        """
        Calculate the population standard deviation of execution times.

        Returns:
            The population standard deviation of execution times in milliseconds.
        """

        if len(self.durations_ms) < 2:  # Standard deviation requires at least two data points.
            return 0.0

        return statistics.pstdev(self.durations_ms)

    @functools.cached_property
    def min_ms(self) -> float:
        """
        Get the minimum execution time in milliseconds.

        Returns:
            The minimum execution time in milliseconds.
        """

        return min(self.durations_ms) if self.durations_ms else 0.0

    @functools.cached_property
    def max_ms(self) -> float:
        """
        Get the maximum execution time in milliseconds.

        Returns:
            The maximum execution time in milliseconds.
        """

        return max(self.durations_ms) if self.durations_ms else 0.0

    def to_dict(self, include_lockers: bool = False) -> BenchmarkResultDict:
        """
        Serialize the benchmark result to a dictionary.

        This is useful for generating structured output like JSON.

        Args:
            include_lockers: If True, include the list of open lockers in the output. This may be a large list.

        Returns:
            A dictionary representation of the benchmark result.
        """

        payload: BenchmarkResultDict = {
            "method_name": self.method_name,
            "count": self.count,
            "repeats": len(self.durations_ms),
            "avg_ms": self.avg_ms,
            "std_ms": self.std_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "durations_ms": self.durations_ms,
        }

        if include_lockers:
            payload["open_lockers"] = self.open_lockers

        return payload


class PerformanceBenchmark:
    """A utility class for benchmarking different Locker Problem solutions."""

    # A reasonable upper limit for O(n log(n)) simulations to prevent excessively long runtimes.
    MAX_SIMULATION_SIZE: int = 50_000

    @staticmethod
    def _measure_execution(func: Callable[..., T], *args, **kwargs) -> tuple[T, float]:
        """
        Measure the execution of a function in milliseconds.

        Uses `time.perf_counter_ns` for high-precision timing.

        Args:
            func: The function to time.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            A tuple containing the function's result and its execution time in milliseconds.
        """

        start_ns: int = time.perf_counter_ns()
        result: T = func(*args, **kwargs)
        end_ns: int = time.perf_counter_ns()

        return result, (end_ns - start_ns) / 1_000_000.0

    @staticmethod
    def print_benchmark_result(result: BenchmarkResult, verbose: bool) -> None:
        """
        Print formatted benchmark results to the console.

        Args:
            result: The `BenchmarkResult` object to print.
            verbose: If True, always print the full list of open lockers.
        """

        print(f"{result.method_name}:")

        # Show locker numbers if requested or if the list is short.
        if verbose or result.count <= 20:
            locker_str: str = ", ".join(map(_format_int, result.open_lockers))
            print(f"  - Open Lockers: {locker_str}")

        print(f"  - Count: {_format_int(result.count)}")

        if len(result.durations_ms) > 1:
            print(
                f"  - Time (ms): avg={result.avg_ms:.3f}, std={result.std_ms:.3f}, "
                f"min={result.min_ms:.3f}, max={result.max_ms:.3f} (over {len(result.durations_ms)} repeats)"
            )
        else:
            print(f"  - Execution Time: {result.avg_ms:.3f}ms")

    @classmethod
    def benchmark_mathematical_solution(cls, num_lockers: int, repeat: int = 1) -> BenchmarkResult:
        """
        Benchmark the mathematical (perfect square) solution.

        Args:
            num_lockers: The total number of lockers.
            repeat: The number of times to run the benchmark for statistical accuracy (must be >= 1).

        Returns:
            A `BenchmarkResult` object with the findings.
        """

        durations: list[float] = []
        last_result: list[int] = []  # We only need to store the result of the last run, as it should be deterministic.

        for _ in range(max(1, repeat)):
            result, exec_time = cls._measure_execution(MathematicalSolution.solve, num_lockers)
            last_result = result

            durations.append(exec_time)

        return BenchmarkResult("Mathematical Solution (O(√n))", last_result, durations)

    @classmethod
    def benchmark_simulation_solution(
        cls, num_lockers: int, show_progress: bool = False, repeat: int = 1
    ) -> BenchmarkResult | None:
        """
        Benchmark the simulation-based solution.

        Args:
            num_lockers: The total number of lockers.
            show_progress: Whether to show progress updates during the simulation.
            repeat: The number of times to run the benchmark for statistical accuracy (must be >= 1).

        Returns:
            A `BenchmarkResult` object with the findings, or None if `num_lockers` exceeds the safety threshold.
        """

        if num_lockers > cls.MAX_SIMULATION_SIZE:
            print(
                f"⚠️ Warning: Simulation for {_format_int(num_lockers)} lockers was skipped because it exceeds the "
                f"configured maximum of {_format_int(cls.MAX_SIMULATION_SIZE)}."
            )

            return None

        def run_simulation(show_p: bool) -> list[int]:
            """
            Encapsulates a single simulation run.

            Args:
                show_p: Whether to show progress updates during the simulation.

            Returns:
                A list of open locker numbers.
            """

            sim: SimulationSolution = SimulationSolution(num_lockers)
            sim.execute_simulation(show_progress=show_p)

            return sim.get_open_lockers()

        durations: list[float] = []
        last_result: list[int] = []

        for i in range(max(1, repeat)):
            # Progress should only be shown on the first run if repeats are requested.
            result, exec_time = cls._measure_execution(run_simulation, show_p=(show_progress and i == 0))

            last_result = result
            durations.append(exec_time)

        return BenchmarkResult("Simulation Solution (O(n log(n)))", last_result, durations)


def _parse_positive_int_arg(text: str) -> int:
    """
    Custom `argparse` type to ensure an argument is a positive integer.

    Also supports values with thousands separators and underscores (e.g., "1,000" or "1_000").

    Args:
        text: The string value from the command-line.

    Returns:
        The converted positive integer.

    Raises:
        argparse.ArgumentTypeError: If `text` is not a positive integer.
    """

    try:
        cleaned: str = text.replace(",", "").replace("_", "").strip()
        value: int = int(cleaned)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid integer value `{text}`.") from e

    if value <= 0:
        raise argparse.ArgumentTypeError(f"Must be a positive integer, but got `{value}`.")

    return value


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the command-line argument parser.

    Returns:
        The configured `ArgumentParser` instance.
    """

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="A Comprehensive Solver For The Classic Locker Problem.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
            %(prog)s                                  # Run with the default settings (1,000 lockers, compare solutions)
            %(prog)s -n 100 --verbose                 # Test 100 lockers and show detailed results
            %(prog)s -n 1000000 --math-only           # Benchmark a large number with the mathematical solution only
            %(prog)s --interactive                    # Enter interactive mode to explore specific lockers or ranges
            %(prog)s -n 10000 --repeat 10 --progress  # Get stable timings by repeating a benchmark 10 times
            %(prog)s -n 100 --json                    # Get machine-readable JSON output for 100 lockers

        For educational purposes:
            %(prog)s -n 25 -v                         # See the full output for a small, easy-to-trace number of lockers
            %(prog)s -i                               # Explore how perfect squares result in open lockers
        """,
    )

    # Core Arguments
    parser.add_argument(
        "-n",
        "--num-lockers",
        type=_parse_positive_int_arg,
        default=1_000,
        metavar="N",
        help="Number of lockers to test (default: %(default)s).",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit.",
    )

    #################### Output Options ####################
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed output, including individual locker numbers.",
    )
    output_group.add_argument(
        "--progress",
        action="store_true",
        help="Show progress updates for long-running simulations.",
    )
    output_group.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the startup banner and headers.",
    )
    output_group.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format (implies `--no-banner`).",
    )
    output_group.add_argument(
        "--json-include-lockers",
        action="store_true",
        help="Include the full list of open locker numbers in JSON output.",
    )

    #################### Solution Selection ####################
    solution_group = parser.add_argument_group("Solution Selection")
    solution_group.add_argument(
        "--math-only",
        action="store_true",
        help="Run only the mathematical solution (fast, recommended for large numbers).",
    )
    solution_group.add_argument(
        "--sim-only",
        action="store_true",
        help="Run only the simulation solution (slower, useful for educational purposes).",
    )

    #################### Special Modes ####################
    mode_group = parser.add_argument_group("Special Modes")
    mode_group.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Run in interactive mode to explore results freely.",
    )

    #################### Benchmark Options ####################
    bench_group = parser.add_argument_group("Benchmark Options")
    bench_group.add_argument(
        "--repeat",
        type=_parse_positive_int_arg,
        default=1,
        metavar="K",
        help="Repeat each benchmark K times and report statistics (default: %(default)s).",
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments and show warnings for problematic combinations.

    Terminates the program if a critical validation error occurs.

    Args:
        args: The parsed command-line arguments from `argparse`.
    """

    if args.math_only and args.sim_only:
        print("❌ Error: Cannot specify both `--math-only` and `--sim-only`.", file=sys.stderr)
        sys.exit(1)

    # Warn the user if they request a simulation that is likely to be very slow.
    if (not args.math_only) and args.num_lockers > PerformanceBenchmark.MAX_SIMULATION_SIZE:
        print(f"⚠️ Warning: Simulation with {_format_int(args.num_lockers)} lockers may be very slow.", file=sys.stderr)

        # In non-interactive sessions (e.g., from a script), fail instead of hanging on input.
        if not sys.stdin.isatty():
            print(
                "Refusing to continue in a non-interactive session. Use `--math-only` to skip the slow simulation.",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            response: str = input("Continue Anyway? (y/N): ")
            if not response.lower().startswith("y"):
                print("Operation cancelled.")
                sys.exit(0)
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)

    # JSON mode implies no banner for clean, machine-readable output.
    if args.json:
        args.no_banner = True


def _build_meta(args: argparse.Namespace) -> dict[str, Any]:
    """
    Build metadata for JSON output.

    Args:
        args: The parsed command-line arguments from `argparse`.

    Returns:
        A dictionary with application and execution metadata.
    """

    return {
        "app": APP_NAME,
        "version": __version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "num_lockers": args.num_lockers,
        "args": vars(args),
    }


def _json_print(payload: dict[str, Any]) -> None:
    """
    Pretty-print a dictionary as a JSON object to stdout.

    Args:
        payload: The dictionary to serialize and print.
    """

    print(json.dumps(payload, indent=2))


def main() -> None:
    """Main entry point for the program. Handles argument parsing, validation, and execution flow."""

    parser: argparse.ArgumentParser = create_argument_parser()
    args: argparse.Namespace = parser.parse_args()

    validate_arguments(args)

    if args.interactive:
        if not args.no_banner:
            _print_banner()

        InteractiveExplorer.run()
        return

    #################### Run Benchmarks ####################
    results: list[BenchmarkResult] = []
    math_result: BenchmarkResult | None = None
    sim_result: BenchmarkResult | None = None

    if not args.sim_only:
        math_result = PerformanceBenchmark.benchmark_mathematical_solution(args.num_lockers, repeat=args.repeat)
        results.append(math_result)
    if not args.math_only:
        sim_result = PerformanceBenchmark.benchmark_simulation_solution(
            args.num_lockers, show_progress=args.progress, repeat=args.repeat
        )
        if sim_result:
            results.append(sim_result)

    #################### Render Results ####################
    if args.json:
        # JSON output
        payload: dict[str, Any] = {
            "meta": _build_meta(args),
            "results": [result.to_dict(include_lockers=args.json_include_lockers) for result in results],
        }

        # Add consistency check if both solutions were run and produced results.
        if math_result and sim_result:
            payload["consistency_check"] = {"match": math_result.open_lockers == sim_result.open_lockers}

        _json_print(payload)
    else:
        # Human-readable console output
        if not args.no_banner:
            _print_banner()

        _print_header(f"Results For {_format_int(args.num_lockers)} Lockers")

        if math_result:
            PerformanceBenchmark.print_benchmark_result(math_result, args.verbose)
            _print_divider()

        if sim_result:
            PerformanceBenchmark.print_benchmark_result(sim_result, args.verbose)
            _print_divider()

        # Cross-check results if both were run.
        if math_result and sim_result:
            is_same: bool = math_result.open_lockers == sim_result.open_lockers
            if is_same:
                print("\n✅ Results match between mathematical and simulation solutions.")
            else:
                print("\n❌ Mismatch detected between solutions!")
                if args.verbose:
                    math_only: set[int] = set(math_result.open_lockers) - set(sim_result.open_lockers)
                    sim_only: set[int] = set(sim_result.open_lockers) - set(math_result.open_lockers)

                    if math_only:
                        print(f"  - Present Only In Mathematical Solution: {sorted(math_only)}")
                    if sim_only:
                        print(f"  - Present Only In Simulation Solution: {sorted(sim_only)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting.")
        sys.exit(0)
