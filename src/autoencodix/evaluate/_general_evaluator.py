from typing import Any, Union, Tuple
import warnings

import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.metrics import get_scorer

from umap import UMAP
from sklearn.manifold import TSNE
from sklearn.base import ClassifierMixin, RegressorMixin


from autoencodix.utils._result import Result
from autoencodix.data._datasetcontainer import DatasetContainer
from autoencodix.base._base_evaluator import BaseEvaluator


class GeneralEvaluator(BaseEvaluator):
    def __init__(self):
        # super().__init__()
        pass

    def evaluate(
        self,
        datasets: DatasetContainer,
        result: Result,
        ml_model_class: ClassifierMixin = linear_model.LogisticRegression(),  # Default is sklearn LogisticRegression
        ml_model_regression: RegressorMixin = linear_model.LinearRegression(),  # Default is sklearn LinearRegression
        params: Union[
            list, str
        ] = "all",  # No default? ... or all params in annotation?
        metric_class: str = "roc_auc_ovo",  # Default is 'roc_auc_ovo' via https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
        metric_regression: str = "r2",  # Default is 'r2'
        reference_methods: list = [],  # Default [], Options are "PCA", "UMAP", "TSNE", "RandomFeature"
        split_type: str = "use-split",  # Default is "use-split", other options: "CV-5", ... "LOOCV"?
    ) -> Result:
        """
        Evaluate the performance of machine learning models on various feature representations and clinical parameters.
        This method performs classification or regression tasks using specified machine learning models on different feature sets (e.g., latent space, PCA, UMAP, TSNE, RandomFeature) and clinical annotation parameters. It supports multiple evaluation strategies, including pre-defined train/valid/test splits, k-fold cross-validation, and leave-one-out cross-validation. The results are aggregated and stored in the provided `result` object.
        Parameters
        ----------
        datasets : DatasetContainer
            An DatasetContainer containing train, valid, and test datasets, each with `sample_ids` and `metadata` (either a DataFrame or a dictionary with a 'paired' key for clinical annotations).
        result : Result
            An Result object to store the evaluation results. Should have an `embedding_evaluation` attribute which updated (typically a DataFrame).
        ml_model_class : ClassifierMixin, optional
            The scikit-learn classifier to use for classification tasks (default: `sklearn.linear_model.LogisticRegression()`).
        ml_model_regression : RegressorMixin, optional
            The scikit-learn regressor to use for regression tasks (default: `sklearn.linear_model.LinearRegression()`).
        params : list or str, optional
            List of clinical annotation columns to evaluate, or "all" to use all columns (default: "all").
        metric_class : str, optional
            Scoring metric for classification tasks (default: "roc_auc_ovo").
        metric_regression : str, optional
            Scoring metric for regression tasks (default: "r2").
        reference_methods : list, optional
            List of feature representations to evaluate (e.g., "PCA", "UMAP", "TSNE", "RandomFeature"). "Latent" is always included (default: []).
        split_type : str, optional
            Evaluation strategy: "use-split" for pre-defined splits, "CV-N" for N-fold cross-validation, or "LOOCV" for leave-one-out cross-validation (default: "use-split").
        Returns
        -------
        Result
            The updated result object with evaluation results stored in `embedding_evaluation`.
        Raises
        ------
        ValueError
            If required annotation data is missing or improperly formatted, or if an unsupported split type is specified.
        Notes
        -----
        - Samples with missing annotation values for a given parameter are excluded from the corresponding evaluation.
        - For "RandomFeature", five random feature sets are evaluated.
        - The method appends results to any existing `embedding_evaluation` in the result object.
        """

        already_warned = False

        df_results = pd.DataFrame()

        reference_methods.append("Latent")

        # ## TODO when x-modalix is ready, check how to adjust evaluation of both latent spaces
        # is_modalix = False # TODO Remove and implement individual Evaluators later

        # if type(result.latentspaces.get(epoch=-1,split="train")) == dict:
        #     # For X-Modalix and others with multiple VAE Latentspaces
        #     reference_methods = [f"{method}_$_{key}" for method in reference_methods for key in result.latentspaces.get(epoch=-1,split="train").keys()]
        #     is_modalix = True
        reference_methods = self._expand_reference_methods(
            reference_methods=reference_methods, result=result
        )

        for task in reference_methods:
            print(f"Perform ML task with feature df: {task}")

            clin_data = self._get_clin_data(datasets)

            if split_type == "use-split":
                # Pandas dataframe with sample_ids and split information
                sample_split = pd.DataFrame(columns=["SAMPLE_ID", "SPLIT"])

                if hasattr(datasets, "train"):
                    if hasattr(datasets.train, "paired_sample_ids"):
                        if datasets.train.paired_sample_ids is not None:
                            sample_ids = datasets.train.paired_sample_ids
                    else:
                        sample_ids = datasets.train.sample_ids
                    sample_split_temp = dict(
                        sample_split,
                        **{
                            "SAMPLE_ID": sample_ids,
                            "SPLIT": ["train"] * len(sample_ids),
                        },
                    )
                    sample_split = pd.concat(
                        [sample_split, pd.DataFrame(sample_split_temp)],
                        axis=0,
                        ignore_index=True,
                    )
                else:
                    raise ValueError(
                        "No training data found. Please provide a valid training dataset."
                    )
                if hasattr(datasets, "valid"):
                    if hasattr(datasets.valid, "paired_sample_ids"):
                        if datasets.valid.paired_sample_ids is not None:
                            sample_ids = datasets.valid.paired_sample_ids
                    else:
                        sample_ids = datasets.valid.sample_ids
                    sample_split_temp = dict(
                        sample_split,
                        **{
                            "SAMPLE_ID": sample_ids,
                            "SPLIT": ["valid"] * len(sample_ids),
                        },
                    )
                    sample_split = pd.concat(
                        [sample_split, pd.DataFrame(sample_split_temp)],
                        axis=0,
                        ignore_index=True,
                    )
                if hasattr(datasets, "test"):
                    if hasattr(datasets.test, "paired_sample_ids"):
                        if datasets.test.paired_sample_ids is not None:
                            sample_ids = datasets.test.paired_sample_ids
                    else:
                        sample_ids = datasets.test.sample_ids
                    sample_split_temp = dict(
                        sample_split,
                        **{
                            "SAMPLE_ID": sample_ids,
                            "SPLIT": ["test"] * len(sample_ids),
                        },
                    )
                    sample_split = pd.concat(
                        [sample_split, pd.DataFrame(sample_split_temp)],
                        axis=0,
                        ignore_index=True,
                    )

                sample_split = sample_split.set_index("SAMPLE_ID", drop=False)

            ## df -> task
            subtask = [task]
            if "RandomFeature" in task:
                subtask = [task + "_R" + str(x) for x in range(1, 6)]
            for sub in subtask:
                print(sub)
                # if is_modalix:
                #     modality = task.split("_$_")[1]
                #     task_xmodal = task.split("_$_")[0]

                #     df = self._load_input_for_ml_xmodal(task_xmodal, datasets, result, modality=modality)
                # else:
                df = self._load_input_for_ml(task, datasets, result)

                if params == "all":
                    params = clin_data.columns.tolist()

                for task_param in params:
                    if "Latent" in task:
                        print(f"Perform ML task for target parameter: {task_param}")
                    ## Check if classification or regression task
                    ml_type = self._get_ml_type(clin_data, task_param)

                    if pd.isna(clin_data[task_param]).sum() > 0:
                        if not already_warned:
                            print(
                                "There are NA values in the annotation file. Samples with missing data will be removed for ML task evaluation."
                            )
                        already_warned = True
                        # logger.warning(clin_data.loc[pd.isna(clin_data[task_param]), task_param])

                        samples_nonna = clin_data.loc[
                            pd.notna(clin_data[task_param]), task_param
                        ].index
                        # print(df)
                        df = df.loc[samples_nonna.intersection(df.index), :]
                        if split_type == "use-split":
                            sample_split = sample_split.loc[
                                samples_nonna.intersection(sample_split.index), :
                            ]
                        # print(sample_split)

                    if ml_type == "classification":
                        metric = metric_class
                        sklearn_ml = ml_model_class

                    if ml_type == "regression":
                        metric = metric_regression
                        sklearn_ml = ml_model_regression

                    if split_type == "use-split":
                        results = self._single_ml_presplit(
                            sample_split=sample_split,
                            df=df,
                            clin_data=clin_data,
                            task_param=task_param,
                            sklearn_ml=sklearn_ml,
                            metric=metric,
                            ml_type=ml_type,
                        )
                    elif split_type.startswith("CV-"):
                        cv_folds = int(split_type.split("-")[1])

                        results = self._single_ml(
                            df=df,
                            clin_data=clin_data,
                            task_param=task_param,
                            sklearn_ml=sklearn_ml,
                            metric=metric,
                            cv_folds=cv_folds,
                        )
                    elif split_type == "LOOCV":
                        # Leave One Out Cross Validation
                        results = self._single_ml(
                            df=df,
                            clin_data=clin_data,
                            task_param=task_param,
                            sklearn_ml=sklearn_ml,
                            metric=metric,
                            cv_folds=len(df),
                        )
                    else:
                        raise ValueError(
                            f"Your split type {split_type} is not supported. Please use 'use-split', 'CV-5', 'LOOCV' or 'CV-N'."
                        )
                    results = self._enrich_results(
                        results=results,
                        sklearn_ml=sklearn_ml,
                        ml_type=ml_type,
                        task=task,
                        sub=sub,
                    )

                    df_results = pd.concat([df_results, results])

        ## Check if embedding_evaluation is empty
        if (
            hasattr(result, "embedding_evaluation")
            and len(result.embedding_evaluation) == 0
        ):
            result.embedding_evaluation = df_results
        else:
            # merge with existing results
            result.embedding_evaluation = pd.concat(
                [result.embedding_evaluation, df_results], axis=0
            )

        return result

    @staticmethod
    def _single_ml(
        df: pd.DataFrame,
        clin_data: pd.DataFrame,
        task_param: str,
        sklearn_ml: Union[ClassifierMixin, RegressorMixin],
        metric: str,
        cv_folds: int = 5,
    ):
        """
        Function learns on the given data frame df and label data the provided sklearn model.
        Cross validation is performed according to the config and scores are returned as output as specified by metrics
        ARGS:
            df (pd.DataFrame): Dataframe with input data
            clin_data (pd.DataFrame): Dataframe with label data
            task_param (str): Column name with label data
            sklearn_ml (sklearn.module): Sklearn ML module specifying the ML algorithm
            metric (str): string specifying the metric to be calculated by cross validation
        RETURNS:
            score_df (pd.DataFrame): data frame containing metrics (scores) for all CV runs (long format)

        """

        # X -> df
        # Y -> task_param
        y = clin_data.loc[df.index, task_param]
        score_df = dict()

        ## Cross Validation
        if len(y.unique()) > 1:
            scores = cross_validate(
                sklearn_ml, df, y, cv=cv_folds, scoring=metric, return_train_score=True
            )

            # Output

            # Output Format
            # CV_RUN | SCORE_SPLIT | TASK_PARAM | METRIC | VALUE

            score_df["cv_run"] = list()
            score_df["score_split"] = list()
            score_df["CLINIC_PARAM"] = list()
            score_df["metric"] = list()
            score_df["value"] = list()

            cv_runs = ["CV_" + str(x) for x in range(1, cv_folds + 1)]
            task_param_cv = [task_param for x in range(1, cv_folds + 1)]

            for m in scores:
                if m.split("_")[0] == "test" or m.split("_")[0] == "train":
                    split_cv = [m.split("_")[0] for x in range(1, cv_folds + 1)]
                    metric_cv = [metric for x in range(1, cv_folds + 1)]

                    score_df["cv_run"].extend(cv_runs)
                    score_df["score_split"].extend(split_cv)
                    score_df["CLINIC_PARAM"].extend(task_param_cv)
                    score_df["metric"].extend(metric_cv)
                    score_df["value"].extend(scores[m])

        return pd.DataFrame(score_df)

    @staticmethod
    def _get_clin_data(datasets) -> pd.DataFrame:
        """
        Retrieves the clinical annotation DataFrame (clin_data) from the provided datasets.
        Handles both standard and XModalix dataset structures.
        """
        if hasattr(datasets.train, "metadata"):
            # Check if metadata is a dictionary and contains 'paired'
            if isinstance(datasets.train.metadata, dict):
                if "paired" in datasets.train.metadata:
                    clin_data = datasets.train.metadata["paired"]
                    if hasattr(datasets, "test"):
                        clin_data = pd.concat(
                            [clin_data, datasets.test.metadata["paired"]],
                            axis=0,
                        )
                    if hasattr(datasets, "valid"):
                        clin_data = pd.concat(
                            [clin_data, datasets.valid.metadata["paired"]],
                            axis=0,
                        )
                else:
                    # Iterate over all splits and keys, concatenate if DataFrame
                    clin_data = pd.DataFrame()
                    for split_name in ["train", "test", "valid"]:
                        split_temp = getattr(datasets, split_name, None)
                        if split_temp is not None and hasattr(split_temp, "metadata"):
                            for key in split_temp.metadata.keys():
                                if isinstance(split_temp.metadata[key], pd.DataFrame):
                                    clin_data = pd.concat(
                                        [
                                            clin_data,
                                            split_temp.metadata[key],
                                        ],
                                        axis=0,
                                    )
                    # remove duplicate rows
                    clin_data = clin_data[~clin_data.index.duplicated(keep="first")]
                    # Raise error no annotation given
                    raise ValueError(
                        "Please provide paired annotation data with key 'paired' in metadata dictionary."
                    )
            elif isinstance(datasets.train.metadata, pd.DataFrame):
                clin_data = datasets.train.metadata
                if hasattr(datasets, "test"):
                    clin_data = pd.concat(
                        [clin_data, datasets.test.metadata],
                        axis=0,
                    )
                if hasattr(datasets, "valid"):
                    clin_data = pd.concat(
                        [clin_data, datasets.valid.metadata],
                        axis=0,
                    )
            else:
                # Raise error no annotation given
                raise ValueError(
                    "Metadata is not a dictionary or DataFrame. Please provide a valid annotation data type."
                )
        # elif hasattr(datasets.train, "datasets"):
        #     # XModalix-Case
        #     clin_data = pd.DataFrame()
        #     splits = [
        #         datasets.train, datasets.valid, datasets.test
        #     ]

        #     for s in splits:
        #         for k in s.datasets.keys():
        #             print(f"Processing dataset: {k}")
        #             # Merge metadata by overlapping columns
        #             overlap = clin_data.columns.intersection(s.datasets[k].metadata.columns)
        #             if overlap.empty:
        #                 overlap = s.datasets[k].metadata.columns
        #             clin_data = pd.concat([clin_data, s.datasets[k].metadata[overlap]], axis=0)

        #     # Remove duplicate rows
        #     clin_data = clin_data[~clin_data.index.duplicated(keep='first')]
        else:
            # Iterate over all splits and keys, concatenate if DataFrame
                clin_data = pd.DataFrame()
                for split_name in ["train", "test", "valid"]:
                    split_temp = getattr(datasets, split_name, None)
                    if split_temp is not None:
                        for key in split_temp.datasets.keys():
                            if isinstance(split_temp.datasets[key].metadata, pd.DataFrame):
                                clin_data = pd.concat(
                                    [
                                        clin_data,
                                        split_temp.datasets[key].metadata,
                                    ],
                                    axis=0,
                                )
                # remove duplicate rows
                clin_data = clin_data[~clin_data.index.duplicated(keep="first")]
            # Raise error no annotation given
            # raise ValueError(
            #     "No annotation data found. Please provide a valid annotation data type."
            # )
        return clin_data

    def _enrich_results(
        self,
        results: pd.DataFrame,
        sklearn_ml: Union[ClassifierMixin, RegressorMixin],
        ml_type: str,
        task: str,
        sub: str,
    ) -> pd.DataFrame:
        res_ml_alg = [str(sklearn_ml) for x in range(0, results.shape[0])]
        res_ml_type = [ml_type for x in range(0, results.shape[0])]
        res_ml_task = [task for x in range(0, results.shape[0])]
        res_ml_subtask = [sub for x in range(0, results.shape[0])]

        results["ML_ALG"] = res_ml_alg
        results["ML_TYPE"] = res_ml_type
        # if is_modalix:
        #     results["MODALITY"] = [modality for x in range(0, results.shape[0])]
        #     results["ML_TASK"] = [task_xmodal for x in range(0, results.shape[0])]
        # else:
        results["ML_TASK"] = res_ml_task
        results["ML_SUBTASK"] = res_ml_subtask

        return results

    @staticmethod
    def _single_ml_presplit(
        sample_split: pd.DataFrame,
        df: pd.DataFrame,
        clin_data: pd.DataFrame,
        task_param: str,
        sklearn_ml: Union[ClassifierMixin, RegressorMixin],
        metric: str,
        ml_type: str,
    ):
        """
        Trains the provided sklearn model on the training split and evaluates it on train, valid, and test splits using the specified metric.

        Parameters
        ----------
        sample_split : pd.DataFrame
            DataFrame with sample IDs and their corresponding split ("train", "valid", "test").
        df : pd.DataFrame
            DataFrame with input features, indexed by sample IDs.
        clin_data : pd.DataFrame
            DataFrame with label/annotation data, indexed by sample IDs.
        task_param : str
            Column name in clin_data specifying the target variable.
        sklearn_ml : ClassifierMixin or RegressorMixin
            Instantiated sklearn model to use for training and evaluation.
        metric : str
            Scoring metric compatible with sklearn's get_scorer.
        ml_type : str
            Type of machine learning task ("classification" or "regression").

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation scores for each split (train, valid, test) and the specified metric.

        Raises
        ------
        ValueError
            If the provided metric is not supported by sklearn.
        """
        split_list = ["train", "valid", "test"]

        score_df = dict()
        score_df["score_split"] = list()
        score_df["CLINIC_PARAM"] = list()
        score_df["metric"] = list()
        score_df["value"] = list()

        X_train = df.loc[
            sample_split.loc[sample_split.SPLIT == "train", "SAMPLE_ID"], :
        ]
        train_samples = [s for s in X_train.index]
        Y_train = clin_data.loc[train_samples, task_param]
        # train model once on training data
        if len(Y_train.unique()) > 1:
            sklearn_ml.fit(X_train, Y_train)

            # eval on all splits
            for split in split_list:
                X = df.loc[
                    sample_split.loc[sample_split.SPLIT == split, "SAMPLE_ID"], :
                ]
                samples = [s for s in X.index]
                Y = clin_data.loc[samples, task_param]

                # Performace on train, valid and test data split

                score_df["score_split"].append(split)
                score_df["CLINIC_PARAM"].append(task_param)
                score_df["metric"].append(metric)
                sklearn_scorer = get_scorer(metric)

                if sklearn_scorer is None:
                    raise ValueError(
                        f"Your metric {metric} is not supported by sklearn. Please use a valid metric."
                    )

                if ml_type == "classification":
                    # Check that Y has only classes which are present in Y_train
                    if len(set(Y.unique()).difference(set(Y_train.unique()))) > 0:
                        print(
                            f"Classes in split {split} are not present in training data"
                        )
                        # Adjust Y to only contain classes present in Y_train
                        Y = Y[Y.isin(Y_train.unique())]
                        # Adjust X as well
                        X = X.loc[Y.index, :]

                score_temp = sklearn_scorer(sklearn_ml, X, Y)
                score_df["value"].append(score_temp)
        else:
            ## Warning that there is only one class in the training data
            warnings.warn(
                f"Warning: There is only one class in the training data for task parameter {task_param}. Skipping evaluation for this task."
            )

        return pd.DataFrame(score_df)

    @staticmethod
    def _get_ml_type(clin_data: pd.DataFrame, task_param: str) -> str:
        """
        Determines the machine learning task type (classification or regression) based on the data type of a specified column in clinical data.

        Args:
            clin_data (pd.DataFrame): The clinical data as a pandas DataFrame.
            task_param (str): The column name in clin_data to inspect for determining the task type.

        Returns:
            str: "classification" if the first value in the specified column is a string, otherwise "regression".
        """
        ## Auto-Detection
        if type(list(clin_data[task_param])[0]) is str:
            ml_type = "classification"
        elif clin_data[task_param].unique().shape[0] < 3:
            ml_type = "classification"
        else:
            ml_type = "regression"

        return ml_type

    @staticmethod
    def _load_input_for_ml(
        task: str, dataset: DatasetContainer, result: Result
    ) -> pd.DataFrame:
        """
        Loads and processes input data for various machine learning tasks based on the specified task type.
        Parameters:
            task (str): The type of ML task. Supported values are "Latent", "UMAP", "PCA", "TSNE", and "RandomFeature".
            dataset (DatasetContainer): The dataset container object holding train, validation, and test splits.
            result (Result): The result object containing model configuration and methods to retrieve latent representations.
        Returns:
            pd.DataFrame: A DataFrame containing the processed input data suitable for the specified ML task.
        Raises:
            ValueError: If the provided task is not supported.
        Task Details:
            - "Latent": Concatenates latent representations from train, validation, and test splits at the final epoch.
            - "UMAP": Applies UMAP dimensionality reduction to the concatenated dataset splits.
            - "PCA": Applies PCA dimensionality reduction to the concatenated dataset splits.
            - "TSNE": Applies t-SNE dimensionality reduction to the concatenated dataset splits.
            - "RandomFeature": Randomly samples columns (features) from the concatenated dataset splits.
        """

        final_epoch = result.model.config.epochs - 1

        if task == "Latent":
            df = pd.concat(
                [
                    result.get_latent_df(epoch=final_epoch, split="train"),
                    result.get_latent_df(epoch=final_epoch, split="valid"),
                    result.get_latent_df(epoch=-1, split="test"),
                ]
            )
        elif task in ["UMAP", "PCA", "TSNE", "RandomFeature"]:
            df_processed = pd.concat(
                [
                    dataset.train._to_df(),
                    dataset.test._to_df(),
                    dataset.valid._to_df(),
                ]
            )
            if task == "UMAP":
                reducer = UMAP(n_components=result.model.config.latent_dim)
                df = pd.DataFrame(
                    reducer.fit_transform(df_processed), index=df_processed.index
                )
            elif task == "PCA":
                reducer = PCA(n_components=result.model.config.latent_dim)
                df = pd.DataFrame(
                    reducer.fit_transform(df_processed), index=df_processed.index
                )
            elif task == "TSNE":
                reducer = TSNE(n_components=result.model.config.latent_dim)
                df = pd.DataFrame(
                    reducer.fit_transform(df_processed), index=df_processed.index
                )
            elif task == "RandomFeature":
                df = df_processed.sample(n=result.model.config.latent_dim, axis=1)
        else:
            raise ValueError(
                f"Your ML task {task} is not supported. Please use Latent, UMAP, PCA or RandomFeature."
            )

        return df
