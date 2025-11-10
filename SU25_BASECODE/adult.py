# Copyright (c) 2023 David Boetius
# Licensed under the MIT license
import itertools
from typing import Callable, Optional, Tuple, Union

import os
from pathlib import Path
import requests
from collections import OrderedDict
import hashlib
from zipfile import ZipFile

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset


def sha256sum(path):
    # based on: https://stackoverflow.com/a/3431838/10550998 by quantumSoup
    # License: CC-BY-SA
    hash_ = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


class Adult(Dataset):
    """
    The `Adult <https://archive.ics.uci.edu/dataset/2/adult>`_ dataset.

    The dataset is preprocessed by:
     - removing rows (samples) with missing values
     - one-hot encoding all categorical attributes
     - applying z-score normalization to all continuous variables

    Attributes:
     - `dataset_url`: The URL the Adult dataset is downloaded from.
     - `files_to_download`: The files that are downloaded from `dataset_url`.
     - `checksums`: The checksums of the files in `files`.
     - `train_file`: The file containing the training data after downloading.
     - `test_file`: The file containing the test data after downloading.
     - `variables_with_values`: The list of variables of the Adult dataset,
        together with the values they may take on.
        For integer variables, such as `age`, the value is :code:`None`.
        That is, :code:`Adult.variables["age"] = None`.
        For categorical variables, like `sex`, the value is a tuple of strings.
        For example, :code:`Adult.variables["sex"] = ("female", "male")`.
        Values that do not appear in the dataset after preprocessing are not
        included.
        This only affects the values of `workclass`, where `workclass=Never-worked`
        is dropped, as it does not appear in the dataset after dropping rows with
        missing values.
     - `columns`: Column labels for the tensors in this dataset (after one-hot encoding).
        This is :code:`Adult.columns = ("age", "workclass=Private",
        "workclass=Self-emp-not-inc", ...)`.
    """

    dataset_url = "https://archive.ics.uci.edu/static/public/2/adult.zip"
    files_to_download = {"test": "adult.test", "train": "adult.data"}
    checksums = {
        "adult.test": "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05",
        "adult.data": "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d",
    }
    train_file = "train.csv"
    test_file = "test.csv"
    variables = OrderedDict(
        [
            ("age", None),  # continuous variables marked with None
            (
                "workclass",
                (
                    "Private",
                    "Self-emp-not-inc",
                    "Self-emp-inc",
                    "Federal-gov",
                    "Local-gov",
                    "State-gov",
                    "Without-pay",
                    # "Never-worked",  # does not appear in dataset
                ),
            ),
            ("fnlwgt", None),
            (
                "education",
                (
                    "Bachelors",
                    "Some-college",
                    "11th",
                    "HS-grad",
                    "Prof-school",
                    "Assoc-acdm",
                    "Assoc-voc",
                    "9th",
                    "7th-8th",
                    "12th",
                    "Masters",
                    "1st-4th",
                    "10th",
                    "Doctorate",
                    "5th-6th",
                    "Preschool",
                ),
            ),
            ("education-num", None),
            (
                "marital-status",
                (
                    "Married-civ-spouse",
                    "Divorced",
                    "Never-married",
                    "Separated",
                    "Widowed",
                    "Married-spouse-absent",
                    "Married-AF-spouse",
                ),
            ),
            (
                "occupation",
                (
                    "Tech-support",
                    "Craft-repair",
                    "Other-service",
                    "Sales",
                    "Exec-managerial",
                    "Prof-specialty",
                    "Handlers-cleaners",
                    "Machine-op-inspct",
                    "Adm-clerical",
                    "Farming-fishing",
                    "Transport-moving",
                    "Priv-house-serv",
                    "Protective-serv",
                    "Armed-Forces",
                ),
            ),
            (
                "relationship",
                (
                    "Wife",
                    "Own-child",
                    "Husband",
                    "Not-in-family",
                    "Other-relative",
                    "Unmarried",
                ),
            ),
            (
                "race",
                ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"),
            ),
            ("sex", ("Female", "Male")),
            ("capital-gain", None),
            ("capital-loss", None),
            ("hours-per-week", None),
            (
                "native-country",
                (
                    "United-States",
                    "Cambodia",
                    "England",
                    "Puerto-Rico",
                    "Canada",
                    "Germany",
                    "Outlying-US(Guam-USVI-etc)",
                    "India",
                    "Japan",
                    "Greece",
                    "South",
                    "China",
                    "Cuba",
                    "Iran",
                    "Honduras",
                    "Philippines",
                    "Italy",
                    "Poland",
                    "Jamaica",
                    "Vietnam",
                    "Mexico",
                    "Portugal",
                    "Ireland",
                    "France",
                    "Dominican-Republic",
                    "Laos",
                    "Ecuador",
                    "Taiwan",
                    "Haiti",
                    "Columbia",
                    "Hungary",
                    "Guatemala",
                    "Nicaragua",
                    "Scotland",
                    "Thailand",
                    "Yugoslavia",
                    "El-Salvador",
                    "Trinadad&Tobago",
                    "Peru",
                    "Hong",
                    "Holand-Netherlands",
                ),
            ),
        ]
    )
    _label_map = {"<=50K": False, ">50K": True, "<=50K.": False, ">50K.": True}

    columns = tuple(itertools.chain(*(
        [col_name] if values is None else [f"{col_name}={value}" for value in values]
        for col_name, values in variables.items()
    )))

    def __init__(
        self,
        root: Union[str, os.PathLike],
        train: bool = True,
        sensitive_attributes: Union[str, Tuple[str, ...]] = ("sex",),
        download: bool = False,
        output_fn: Optional[Callable[[str], None]] = print,
    ):
        """
        Loads the `Adult <https://archive.ics.uci.edu/dataset/2/adult>`_ dataset.

        :param root: The root directory where the Adult folder is placed or
          is to be downloaded to if download is set to True.
        :param train: Whether to retrieve the training set or test set of
          the dataset.
        :param sensitive_attributes: The attributes to consider protected
          in the Adult dataset.
          Examples: :code:`('sex',)`, :code:`('race',)`, or :code:`('sex', 'race')`.
          Instead of a singleton tuple you may also pass a single string.
          The column indices to which these attributes correspond to
          are be accessible via the property :code:`protected_column_indices`.
          Due to the one-hot encoding of the data, more column indices are
          in that property than you pass in here.
        :param download: Whether to download the Adult dataset from
          https://archive.ics.uci.edu/dataset/2/adult if it is not
          present in the root directory.
        :param output_fn: A function for producing command line output.
          For example, :code:`print` or :code:`logging.info`.
          Pass `None` to turn off command line output.
        """
        if output_fn is None:

            def do_nothing(_):
                pass

            self.__output_fn = do_nothing
        else:
            self.__output_fn = output_fn

        self.files_dir = Path(root, type(self).__name__)
        if not self.files_dir.exists():
            if not download:
                raise RuntimeError(
                    "Dataset not found. Download it by passing download=True."
                )
            os.makedirs(self.files_dir)
            train_raw, test_raw = self._download()
            train_data, test_data = self._preprocess(train_raw, test_raw)

            # create new csv files for the transformed data
            train_data.to_csv(Path(self.files_dir, self.train_file), index=False)
            test_data.to_csv(Path(self.files_dir, self.test_file), index=False)

            if train:
                table = train_data
            else:
                table = test_data
        else:
            if train:
                table = pandas.read_csv(Path(self.files_dir, self.train_file))
            else:
                table = pandas.read_csv(Path(self.files_dir, self.test_file))

        data = table.drop("income", axis=1)
        targets = table["income"]

        if isinstance(sensitive_attributes, str):
            sensitive_attributes = (sensitive_attributes,)
        # get all columns that start with a protected attribute
        self.__sensitive_colum_indices = tuple(
            data.columns.get_loc(col)
            for col in table.columns
            if any(col.startswith(att) for att in sensitive_attributes)
        )

        self.data = torch.tensor(
            data.values.astype(np.float64), dtype=torch.get_default_dtype()
        )
        self.targets = torch.tensor(targets.values.astype(np.int64))

    def _output(self, message: str):
        self.__output_fn(message)

    def _download(self) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        Downloads the Adult dataset, extracts the data and returns
        the raw training data and the raw test data as pandas :code:`DataFrames`.
        """
        self._output("Downloading Adult dataset file...")
        dataset_path = self.files_dir / "dataset.zip"
        try:
            dataset_path.touch(exist_ok=False)
            result = requests.get(self.dataset_url, stream=True)
            with open(dataset_path, "wb") as dataset_file:
                for chunk in result.iter_content(chunk_size=256):
                    dataset_file.write(chunk)
            with ZipFile(dataset_path) as dataset_archive:
                for file_name in self.files_to_download.values():
                    dataset_archive.extract(file_name, self.files_dir)
        finally:
            dataset_path.unlink(missing_ok=True)

        self._output("Checking integrity of downloaded files...")
        for file_name, checksum in self.checksums.items():
            file_path = Path(self.files_dir, file_name)
            downloaded_file_checksum = sha256sum(file_path)
            if checksum != downloaded_file_checksum:
                raise RuntimeError(
                    f"Downloaded file has different checksum than expected: {file_name}. "
                    f"Expected sha256 checksum: {checksum}"
                )
        self._output("Download finished.")

        all_columns = list(self.variables.keys()) + ["income"]
        train_data: pandas.DataFrame = pandas.read_csv(
            Path(self.files_dir, self.files_to_download["train"]),
            header=None,
            index_col=False,
            names=all_columns,
        )
        test_data: pandas.DataFrame = pandas.read_csv(
            Path(self.files_dir, self.files_to_download["test"]),
            header=0,  # first colum contains a note that we throw away
            index_col=False,
            names=all_columns,
        )
        return train_data, test_data

    def _preprocess(
        self, train_raw: pandas.DataFrame, test_raw: pandas.DataFrame
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        self._output("Preprocessing data...")
        # code closely follows:
        # https://github.com/eth-sri/lcifr/blob/master/code/datasets/adult.py
        train_data = self._strip_strings(train_raw)
        test_data = self._strip_strings(test_raw)

        # transform the dataset: drop rows with missing values
        # missing values are encoded as ? in the original tables
        self._remove_missing_values(train_data)
        self._remove_missing_values(test_data)

        # map labels to (uniform) boolean values
        self._preprocess_labels(train_data)
        self._preprocess_labels(test_data)

        train_data, test_data = self._preprocess_features(train_data, test_data)
        self._output("Preprocessing finished.")
        return train_data, test_data

    @staticmethod
    def _strip_strings(table: pandas.DataFrame):
        """Strips all strings in a :code:`DataFrame`."""
        # preprocess strings
        return table.map(lambda val: val.strip() if isinstance(val, str) else val)

    @staticmethod
    def _remove_missing_values(table: pandas.DataFrame):
        """
        Removes rows with missing values (encoded as '?') from table.
        Modifies :code:`table` in-place.
        """
        table.replace(to_replace="?", value=np.nan, inplace=True)
        table.dropna(axis=0, inplace=True)

    def _preprocess_labels(self, table: pandas.DataFrame):
        """
        Applies :code:`self.label_map`.
        Modifies :code:`table` in-place.
        """
        table.replace(self._label_map, inplace=True)

    def _preprocess_features(
        self, train_data: pandas.DataFrame, test_data: pandas.DataFrame
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
        """
        Applies a one-hot encoding to all categorical attributes and
        z-score normalization to all continuous attributes.
        """
        # one-hot encode all categorical variables
        categorical_cols = [
            col for col, vals in self.variables.items() if vals is not None
        ]
        train_data = pandas.get_dummies(
            train_data, columns=categorical_cols, prefix_sep="="
        )
        test_data = pandas.get_dummies(
            test_data, columns=categorical_cols, prefix_sep="="
        )

        # The test data does not contain people of Dutch origin, add the column explicitly
        # Both datasets lack people who have never worked.
        for col in self.columns:
            if col not in train_data.columns:
                train_data.insert(loc=0, column=col, value=0.0)
            if col not in test_data.columns:
                test_data.insert(loc=0, column=col, value=0.0)

        # reorder dataset columns
        all_columns = list(self.columns) + ["income"]
        train_data = train_data[all_columns]
        test_data = test_data[all_columns]

        # standardise continuous columns (z score)
        continuous_cols = [
            col for col, vals in self.variables.items() if vals is None
        ]
        for col in continuous_cols:
            mean = train_data[col].mean()
            std = train_data[col].std()
            train_data[col] = (train_data[col] - mean) / std
            test_data[col] = (test_data[col] - mean) / std
        return train_data, test_data

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

    @property
    def sensitive_column_indices(self):
        """
        The columns in the data that correspond to protected attributes
        """
        return self.__sensitive_colum_indices
