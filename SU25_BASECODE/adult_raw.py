# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
from typing import Tuple
from functools import reduce

import pandas

from .adult import Adult


class AdultRaw(Adult):
    """
    The `Adult <https://archive.ics.uci.edu/dataset/2/adult>`_ dataset.

    In difference, to the :code:`Adult` class, this class does not
    one-hot encode categorical attributes or normalize the continuous
    variables.
    However, it also removes rows with missing values.
    The categorical variables are instead encoded as integers.

    Attributes:
    The attributes are as for :code:`Adult`, except that the value
    of `AdultRaw.columns` has a different value.
    In particular, :code:`AdultRaw.columns = ("age", "workclass", "fnlwgt", ...)`
    """

    _categorical_features_map = reduce(
        lambda d1, d2: d1 | d2,
        (
            {value: index for index, value in enumerate(values)}
            for values in Adult.variables.values()
            if values is not None
        ),
    )

    columns = tuple(col_name for col_name in Adult.variables)

    def _preprocess_features(
        self, train_data: pandas.DataFrame, test_data: pandas.DataFrame
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        Replaces string values of categorical attributes with integers.
        """
        for table in [train_data, test_data]:
            table.replace(self._categorical_features_map, inplace=True)

        all_columns = list(self.columns) + ["income"]
        train_data = train_data[all_columns]
        test_data = test_data[all_columns]
        return train_data, test_data
