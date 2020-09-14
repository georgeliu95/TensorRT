import argparse

from polygraphy.tools.util import args as args_util

class Tool(object):
    def __init__(self):
        self.__doc__ = None
        self.name = None


    def add_parser_args(self, parser):
        pass


    def setup_parser(self, subparsers):
        add_help = True
        tool_help = self.__doc__ if self.__doc__ is not None else ""
        if not tool_help.strip():
            add_help = False
            tool_help = "<No help output provided>"

        parser = subparsers.add_parser(self.name, help=tool_help, add_help=add_help, description=tool_help)
        parser.set_defaults(subcommand=self)

        self.add_parser_args(parser)
        args_util.add_logger_args(parser)


    def __call__(self, args):
        raise NotImplementedError("Tool is an abstract class. Please implement this function for your tool")
