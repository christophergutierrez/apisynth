"""This file should be excluded from scanning."""


def this_should_not_appear():
    pass


class AlsoExcluded:
    def hidden_method(self):
        pass
