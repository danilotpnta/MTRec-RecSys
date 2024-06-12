from polars import DataFrame

class PolarsDataFrameWrapper:
    """DataFrame wrapper for Polars DataFrame to enable slicing with step and make it interoperable with torch.Dataset objects."""

    def __init__(self, dataframe: DataFrame):
        self.data = dataframe

    def __getitem__(self, key):
        if isinstance(key, int):
            # Handle negative indexing
            if key < 0:
                key += self.data.height
            if key < 0 or key >= self.data.height:
                raise IndexError("Index out of bounds")
            # Fetch a single row as a DataFrame
            return self.data.slice(key, 1)
        elif isinstance(key, slice):
            start, stop, step = key.start, key.stop, key.step
            # Adjust for negative indexing and None values
            if start is None:
                start = 0
            elif start < 0:
                start += self.data.height
            if stop is None:
                stop = self.data.height
            elif stop < 0:
                stop += self.data.height
            # Calculate length for slice
            length = stop - start
            if step is None or step == 1:
                return self.data.slice(start, length)
            else:
                # For steps other than 1, use take with a list of indices
                indices = range(start, stop, step)
                return self.data[indices]
        else:
            raise TypeError("Invalid argument type.")

