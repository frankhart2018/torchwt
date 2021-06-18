from json import load


from collections import namedtuple


class LoaderParser:
    @staticmethod
    def parse_loader(loader_dict):
        loader_args = loader_dict.pop('args')

        batch_size = loader_args.pop('batch_size')

        loader_spec_tuple = namedtuple("LoaderSpecTuple", ["batch_size"])
        loader_spec = loader_spec_tuple(batch_size=batch_size)

        return loader_spec