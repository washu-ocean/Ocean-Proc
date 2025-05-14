import argparse
import json
import sys



class OceanParser(argparse.ArgumentParser):

    def convert_arg_line_to_args(self, arg_line: str) -> list[str]:
        """
        Overridded method of argparse.ArgumentParser class.

        Enables the ability to list the argument flag and its value on the same line
        when reading arguments from a file. Multiple arguments and their values can 
        also be on the same line.

        :param arg_line: a line of text read from a file
        :type arg_line: str
        :return: list of arguments and their values parsed from the 'arg_line' argument
        :rtype: list[str]
        """
        return arg_line.strip().split()
        # return super().convert_arg_line_to_args(arg_line)
    

    def _read_args_from_files(self, arg_strings:list[str]):
        """
        Overridded method of argparse.ArgumentParser class.

        Reads in each argument string from the input list and checks if any 
        string is a reference to an argument-file by checking if the value of 
        the first character is equal to self.fromfile_prefix_chars. If so, 
        attempts to open this string as a file, and read its contents into the
        the new_arg_strings list as new arguments. The abilities to read in JSON 
        files, along with file arguments being prepended to the list have been
        added to this function

        :param arg_strings: as list of arguments read in from the command line
        :type arg_strings: list[str]
        :return: list of old and new (if any) arguments and their values
        :rtype: list[str]
        """
        # expand arguments referencing files
        new_arg_strings = []
        for arg_string in arg_strings:

            # for regular arguments, just add them back into the list
            if not arg_string or arg_string[0] not in self.fromfile_prefix_chars:
                new_arg_strings.append(arg_string)

            # replace arguments referencing files with the file content
            else:
                try:
                    with open(arg_string[1:]) as args_file:
                        file_arg_strings = []
                        if arg_string.endswith(".json"):
                            jd = json.load(args_file)
                            for k,v in jd.items():
                                file_arg_strings.append(k)
                                if isinstance(v, list):
                                    for val in v:
                                        file_arg_strings.append(str(val))
                                elif isinstance(v, str):
                                    if not v or not v.strip():
                                        continue
                                    else:
                                        file_arg_strings.append(v)
                                else:
                                    file_arg_strings.append(str(v))
                        else:
                            for arg_line in args_file.read().splitlines():
                                for arg in self.convert_arg_line_to_args(arg_line):
                                    file_arg_strings.append(arg)
                        file_arg_strings = self._read_args_from_files(file_arg_strings)
                        new_arg_strings = file_arg_strings + new_arg_strings
                except OSError:
                    err = sys.exc_info()[1]
                    self.error(str(err))
        # return the modified argument list
        return new_arg_strings
    
    def print_usage(self, file=None):
        """
        Overridded method of argparse.ArgumentParser class.

        Prints the usage message for the argparse.ArgumentParser object.
        An example format for an arguments file input as been appended

        :param file: a file to print this usage message to (defaults to sys.stdout if None)
        :type file: IO[str]
        """
        if file is None:
            file = sys.stdout
        self._print_message(self.format_usage().rstrip("\n") + "  @[ARG_FILE]\n", 
                            file)

class NoOverride(argparse.Action):
    """
        Custom class that inherits from the argparse.Action class.

        Creates a custom action for an argument that doesn't allow namespace
        values to be overridden by later inputs
        """
    def __call__(self, parser, namespace, values, option_string=None):
        if getattr(namespace, self.dest, None) == None:
                setattr(namespace, self.dest, values)