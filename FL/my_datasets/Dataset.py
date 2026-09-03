class Dataset:
    """
    This class defines the Dataset API.
    All datasets must follow this API.

    """

    def __init__(
        self,
        size: int,
        args: dict
    ):
       self.size = size
       self.args = args

    def fetch(self, client_index):
        """
        Function to get the datasets of given client

        Returns
        -------
        torch.utils.Dataset(decentralizepy.datasets.Data)

        Raises
        ------
        RuntimeError
            If the training set was not initialized

        """
        raise NotImplementedError

class Data:
    """
    This class defines the API for Data.

    """

    def __init__(self, x, y):
        """
        Constructor

        Parameters
        ----------
        x : numpy array
            A numpy array of data samples
        y : numpy array
            A numpy array of outputs corresponding to the sample

        """
        self.x = x
        self.y = y

    def __len__(self):
        """
        Return the number of samples in the dataset

        Returns
        -------
        int
            Number of samples

        """
        return self.y.shape[0]

    def __getitem__(self, i):
        """
        Function to get the item with index i.

        Parameters
        ----------
        i : int
            Index

        Returns
        -------
        2-tuple
            A tuple of the ith data sample and it's corresponding label

        """
        return self.x[i], self.y[i]
