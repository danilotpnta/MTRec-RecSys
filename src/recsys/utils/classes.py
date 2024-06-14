from polars import DataFrame

class PolarsDataFrameWrapper:
    """DataFrame wrapper for Polars DataFrame to enable slicing with step and make it interoperable with torch.Dataset objects."""

    dataframe: DataFrame
    def __init__(self, dataframe: DataFrame):
        self.dataframe = dataframe

    def __getitem__(self, key):
        if isinstance(key, int):
            # Handle negative indexing
            if key < 0:
                key += self.dataframe.height
            if key < 0 or key >= self.dataframe.height:
                raise IndexError("Index out of bounds")
            # Fetch a single row as a dataframeFrame
            return self.dataframe.slice(key, 1)
        elif isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            # Adjust for negative indexing and None values
            if start is None:
                start = 0
            elif start < 0:
                start += self.dataframe.height
            if stop is None:
                stop = self.dataframe.height
            elif stop < 0:
                stop += self.dataframe.height
            # Calculate length for slice
            length = stop - start
            if step is None or step == 1:
                return self.dataframe.slice(start, length)
            else:
                # For steps other than 1, use take with a list of indices
                indices = range(start, stop, step)
                return self.dataframe[indices]
        else:
            raise TypeError("Invalid argument type.")

