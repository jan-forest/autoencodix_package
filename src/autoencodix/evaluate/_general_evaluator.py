from typing import Union, no_type_check
import warnings

import numpy as np
import pandas as pd
import sklearn
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
from autoencodix.base._base_visualizer import BaseVisualizer

sklearn.set_config(enable_metadata_routing=True)


class GeneralEvaluator(BaseEvaluator):
    def __init__(self):
        # super().__init__()
        pass

    @no_type_check
    def evaluate(
        self,
        datasets: DatasetContainer,
        result: Result,
        ml_model_class: ClassifierMixin = linear_model.LogisticRegression(max_iter=1000),  # Default is sklearn LogisticRegression
        ml_model_regression: RegressorMixin = linear_model.LinearRegression(),  # Default is sklearn LinearRegression
        params: Union[
            list, str
        ] = "all",  # No default? ... or all params in annotation?
        metric_class: str = "roc_auc_ovo",  # Default is 'roc_auc_ovo' via https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
        metric_regression: str = "r2",  # Default is 'r2'
        reference_methods: list = [],  # Default [], Options are "PCA", "UMAP", "TSNE", "RandomFeature"
        split_type: str = "use-split",  # Default is "use-split", other options: "CV-5", ... "LOOCV"?
        n_downsample: Union[
            int, None
        ] = 10000,  # Default is 10000, if provided downsample to this number of samples for faster evaluation. Set to None to disable downsampling.
    ) -> Result:
        """Evaluates the performance of machine learning models on various feature representations and clinical parameters.

        This method performs classification or regression tasks using specified machine learning models on different feature sets (e.g., latent space, PCA, UMAP, TSNE, RandomFeature) and clinical annotation parameters. It supports multiple evaluation strategies, including pre-defined train/valid/test splits, k-fold cross-validation, and leave-one-out cross-validation. The results are aggregated and stored in the provided `result` object.
        - Samples with missing annotation values for a given parameter are excluded from the corresponding evaluation.
        - For "RandomFeature", five random feature sets are evaluated.
        - The method appends results to any existing `embedding_evaluation` in the result object.

        Args:
            datasets: A DatasetContainer containing train, valid, and test datasets, each with `sample_ids` and `metadata` (either a DataFrame or a dictionary with a 'paired' key for clinical annotations).
            result: An Result object to store the evaluation results. Should have an `embedding_evaluation` attribute which updated (typically a DataFrame).
            ml_model_class: The scikit-learn classifier to use for classification tasks (default: `sklearn.linear_model.LogisticRegression()`).
            ml_model_regression: The scikit-learn regressor to use for regression tasks (default: `sklearn.linear_model.LinearRegression()`).
            params:List of clinical annotation columns to evaluate, or "all" to use all columns (default: "all").
            metric_class: Scoring metric for classification tasks (default: "roc_auc_ovo").
            metric_regression: Scoring metric for regression tasks (default: "r2").
            reference_methods:List of feature representations to evaluate (e.g., "PCA", "UMAP", "TSNE", "RandomFeature"). "Latent" is always included (default: []).
            split_type: which split to use
                use-split" for pre-defined splits, "CV-N" for N-fold cross-validation, or "LOOCV" for leave-one-out cross-validation (default: "use-split").
            n_downsample: If provided, downsample the data to this number of samples for faster evaluation. Default is 10000. Set to None to disable downsampling.
        Returns:
            The updated result object with evaluation results stored in `embedding_evaluation`.
        Raises
            ValueError: If required annotation data is missing or improperly formatted, or if an unsupported split type is specified.

        """

        already_warned = False

        df_results = pd.DataFrame()

        reference_methods.append("Latent")

        reference_methods = self._expand_reference_methods(
            reference_methods=reference_methods, result=result
        )

        ## Overwrite original datasets with new_datasets if available after predict with other data
        if datasets is None:
            datasets = DatasetContainer()

        if bool(result.new_datasets.test):
            datasets.test = result.new_datasets.test

        if not bool(datasets.train or datasets.valid or datasets.test):
            raise ValueError(
                "No datasets found in result object. Please run predict with new data or save/load with all datasets by using save_all=True."
            )
        elif split_type == "use-split" and not bool(datasets.train):
            warnings.warn(
                "Warning: No train split found in result datasets for 'use-split' evaluation. ML model cannot be trained without a train split. Switch to cross-validation (CV-5) instead."
            )
            split_type = "CV-5"

        for task in reference_methods:
            print(f"Perform ML task with feature df: {task}")

            # clin_data = self._get_clin_data(datasets)
            clin_data = BaseVisualizer._collect_all_metadata(result=result)
            

            if split_type == "use-split":
                # Pandas dataframe with sample_ids and split information
                sample_split = pd.DataFrame(columns=["SAMPLE_ID", "SPLIT"])

                if datasets.train is not None:
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
                # else:
                #     raise ValueError(
                #         "No training data found. Please provide a valid training dataset."
                #     )
                if datasets.valid is not None:
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
                if datasets.test is not None:
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
                        # if pd.isna(clin_data[task_param]).values.any():
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

                    if n_downsample is not None:
                        if df.shape[0] > n_downsample:
                            sample_idx = np.random.choice(
                                df.shape[0], n_downsample, replace=False
                            )
                            df = df.iloc[sample_idx]
                            if split_type == "use-split":
                                sample_split = sample_split.loc[df.index, :]

                    if ml_type == "classification":
                        metric = metric_class
                        sklearn_ml = ml_model_class

                    if ml_type == "regression":
                        metric = metric_regression
                        sklearn_ml = ml_model_regression

                    if split_type == "use-split":
                        # print("Sample Split:")
                        # print(sample_split)
                        # print("Latent:")
                        # print(df)
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
        """Function learns on the given data frame df and label data the provided sklearn model.

        Cross validation is performed according to the config and scores are returned as output as specified by metrics

        Args:
            df: Dataframe with input data
            clin_data: Dataframe with label data
            task_param: Column name with label data
            sklearn_ml: Sklearn ML module specifying the ML algorithm
            metric: string specifying the metric to be calculated by cross validation
            cv_folds:
        Returns:
            score_df: data frame containing metrics (scores) for all CV runs (long format)

        """

        # X -> df
        # Y -> task_param
        y: Union[pd.Series, pd.DataFrame] = clin_data.loc[df.index, task_param]
        score_df = dict()

        ## Cross Validation
        if len(y.unique()) > 1:  # ty: ignore
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
        """Trains the provided sklearn model on the training split and evaluates it on train, valid, and test splits using the specified metric.

        Args:
            sample_split: DataFrame with sample IDs and their corresponding split ("train", "valid", "test").
            df: DataFrame with input features, indexed by sample IDs.
            clin_data: DataFrame with label/annotation data, indexed by sample IDs.
            task_param: Column name in clin_data specifying the target variable.
            sklearn_ml: Instantiated sklearn model to use for training and evaluation.
            metric: Scoring metric compatible with sklearn's get_scorer.
            ml_type: Type of machine learning task ("classification" or "regression").

        Returns:
            DataFrame containing evaluation scores for each split (train, valid, test) and the specified metric.

        Raises
            ValueError: If the provided metric is not supported by sklearn.
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
        if len(Y_train.unique()) > 1:  # ty: ignore
            sklearn_ml.fit(X_train, Y_train)  # ty: ignore

            # eval on all splits
            for split in split_list:
                X = df.loc[
                    sample_split.loc[sample_split.SPLIT == split, "SAMPLE_ID"], :
                ]
                if X.shape[0] == 0:
                    # No samples in this split, skip
                    continue
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
                    if (
                        len(
                            set(Y.unique()).difference(  # ty: ignore
                                set(Y_train.unique())  # ty: ignore
                            )  # ty: ignore
                        )  # ty: ignore
                        > 0  # ty: ignore
                    ):  # ty: ignore
                        print(
                            f"Classes in split {split} are not present in training data"
                        )
                        # Adjust Y to only contain classes present in Y_train
                        Y = Y[Y.isin(Y_train.unique())]  # ty: ignore
                        # Adjust X as well
                        X = X.loc[Y.index, :]

                if ml_type == "classification":
                    score_temp = sklearn_scorer(
                        sklearn_ml, X, Y, labels=np.sort(Y_train.unique())
                    )
                elif ml_type == "regression":
                    score_temp = sklearn_scorer(sklearn_ml, X, Y)
                else:
                    raise ValueError(
                        f"Your ML type {ml_type} is not supported. Please use 'classification' or 'regression'."
                    )
                score_df["value"].append(score_temp)
        else:
            ## Warning that there is only one class in the training data
            warnings.warn(
                f"Warning: There is only one class in the training data for task parameter {task_param}. Skipping evaluation for this task."
            )

        return pd.DataFrame(score_df)

    @staticmethod
    def _get_ml_type(clin_data: pd.DataFrame, task_param: str) -> str:
        """Determines the machine learning task type (classification or regression) based on the data type of a specified column in clinical data.

        Args:
            clin_data: The clinical data as a pandas DataFrame.
            task_param: The column name in clin_data to inspect for determining the task type.

        Returns:
            "classification" if the first value in the specified column is a string, otherwise "regression".
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
        """Loads and processes input data for various machine learning tasks based on the specified task type.


        Task Details:
            - "Latent": Concatenates latent representations from train, validation, and test splits at the final epoch.
            - "UMAP": Applies UMAP dimensionality reduction to the concatenated dataset splits.
            - "PCA": Applies PCA dimensionality reduction to the concatenated dataset splits.
            - "TSNE": Applies t-SNE dimensionality reduction to the concatenated dataset splits.
            - "RandomFeature": Randomly samples columns (features) from the concatenated dataset splits.

        Args:
            task: The type of ML task. Supported values are "Latent", "UMAP", "PCA", "TSNE", and "RandomFeature".
            dataset: The dataset container object holding train, validation, and test splits.
            result: The result object containing model configuration and methods to retrieve latent representations.
        Returns:
            A DataFrame containing the processed input data suitable for the specified ML task.
        Raises:
            ValueError: If the provided task is not supported.
        """

        final_epoch = result.model.config.epochs - 1

        # if task == "Latent":
        #     df = pd.concat(
        #         [
        #             result.get_latent_df(epoch=final_epoch, split="train"),
        #             result.get_latent_df(epoch=final_epoch, split="valid"),
        #             result.get_latent_df(epoch=-1, split="test"),
        #         ]
        #     )

        if task == "Latent":
            dfs = []
            for split in ["train", "valid", "test"]:
                df_split = result.get_latent_df(
                    epoch=final_epoch if split != "test" else -1, split=split
                )
                if df_split is not None and not df_split.empty:
                    dfs.append(df_split)

            df = pd.concat(dfs) if dfs else pd.DataFrame()

        elif task in ["UMAP", "PCA", "TSNE", "RandomFeature"]:
            dfs = []
            for split_name in ["train", "valid", "test"]:
                split_data = getattr(dataset, split_name, None)
                if split_data is not None:
                    dfs.append(split_data._to_df())

            if not dfs:
                raise ValueError(
                    "No available dataset splits (train, valid, test) to process."
                )

            df_processed = pd.concat(dfs)

            # elif task in ["UMAP", "PCA", "TSNE", "RandomFeature"]:
            #     if dataset.train is None:
            #         raise ValueError("train attribute of dataset cannot be None")
            #     if dataset.valid is None:
            #         raise ValueError("valid attribute of dataset cannot be None")
            #     if dataset.test is None:
            #         raise ValueError("test attribute of dataset cannot be None")

            #     df_processed = pd.concat(
            #         [
            #             dataset.train._to_df(),
            #             dataset.test._to_df(),
            #             dataset.valid._to_df(),
            #         ]
            #     )
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
