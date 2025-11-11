"""Abstract class to define a runner."""

from abc import ABC, abstractmethod


class Runner(ABC):
    """Abstract class to define a runner."""

    def __init__(self, commands: list[str] | str | None):
        self.commands = commands

    @property
    def commands(self) -> list[str]:
        """Get the commands to run."""
        return self._commands

    @commands.setter
    def commands(self, commands: list[str] | str | None) -> None:
        """Set the commands to run."""
        if commands is None:
            self._commands = []
        elif isinstance(commands, str):
            self._commands = [commands]
        else:
            self._commands = commands

    @abstractmethod
    def run(self):
        """Run the commands."""
