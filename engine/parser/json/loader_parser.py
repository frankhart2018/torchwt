from json import load


from collections import namedtuple


class LoaderParser:
    @staticmethod
    def parse_loader(loader_dict):
        loader_args = loader_dict.pop('args')

        batch_size = loader_args.pop('batch_size')
        shuffle = loader_args.pop('shuffle')
        split_ratio = loader_args.pop('split_ratio')

        loader_spec_tuple = namedtuple("LoaderSpecTuple", ["batch_size", "shuffle", "split_ratio"])
        loader_spec = loader_spec_tuple(batch_size=batch_size, shuffle=shuffle, split_ratio=split_ratio)

        return loader_spec