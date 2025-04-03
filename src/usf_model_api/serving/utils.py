from typing import Type, ByteString
import pickle


def pickle_deserialize(cls: Type, data: ByteString):
    obj = pickle.loads(data)
    if not isinstance(obj, cls):
        raise ValueError(f"Object is not an instance of {cls}.")

    return obj
