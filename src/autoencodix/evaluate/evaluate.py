from typing import Any, Union

import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.metrics import get_scorer

from umap import UMAP
from sklearn.manifold import TSNE

from autoencodix.utils._result import Result


class Evaluator:
    def __init__(self):
        pass

    def evaluate(self,
                datasets,
                result,
                ml_model_class: Any = linear_model.LogisticRegression(), # Default is sklearn LogisticRegression
                ml_model_regression: Any = linear_model.LinearRegression(), # Default is sklearn LinearRegression
                params: Union[list, str]= "all",	# No default? ... or all params in annotation?
                metric_class: str = "roc_auc_ovr", # Default is 'roc_auc_ovr' via https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names
                metric_regression: str = "r2", # Default is 'r2'
                reference_methods: list = [], # Default [], Options are "PCA", "UMAP", "TSNE", "RandomFeature"
                split_type:str = "use-split", # Default is "use-split", other options: "CV-5", ... "LOOCV"?
                ) -> Result:
        # 'result = Result()'

        already_warned = False

        df_results = pd.DataFrame()
        # if cfg["MODEL_TYPE"] == "x-modalix":
        #     if "Latent" in cfg["ML_TASKS"]:
        #         cfg["ML_TASKS"].remove("Latent")
        #         cfg["ML_TASKS"].append("Latent_FROM")
        #         cfg["ML_TASKS"].append("Latent_TO")
        #         cfg["ML_TASKS"].append("Latent_BOTH")

        reference_methods.append("Latent")

        for task in reference_methods:
            print(f"Perform ML task with feature df: {task}")

            ## Getting clin_data
            if hasattr(datasets.train, "metadata"):
                # Check if metadata is a dictionary and contains 'paired'
                if isinstance(datasets.train.metadata, dict):
                    if "paired" in datasets.train.metadata:
                        clin_data = datasets.train.metadata['paired']
                        if hasattr(datasets, "test"):
                            clin_data = pd.concat(
                                [clin_data, datasets.test.metadata['paired']],
                                axis=0,
                            )
                        if hasattr(datasets, "valid"):
                            clin_data = pd.concat(
                                [clin_data, datasets.valid.metadata['paired']],
                                axis=0,
                            )
                    else:
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
            else:
                # Raise error no annotation given
                raise ValueError(
                    "No annotation data found. Please provide a valid annotation data type."
                )
            

            if split_type == "use-split":
                # Pandas dataframe with sample_ids and split information
                sample_split = pd.DataFrame(columns=["SAMPLE_ID", "SPLIT"])

                if hasattr(datasets, "train"):
                    sample_split_temp = dict(
                        sample_split,
                        **{
                            "SAMPLE_ID": datasets.train.sample_ids,
                            "SPLIT": ["train"] * len(datasets.train.sample_ids),
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
                    sample_split_temp = dict(
                        sample_split,
                        **{
                            "SAMPLE_ID": datasets.valid.sample_ids,
                            "SPLIT": ["valid"] * len(datasets.valid.sample_ids),
                        },
                    )
                    sample_split = pd.concat(
                        [sample_split, pd.DataFrame(sample_split_temp)],
                        axis=0,
                        ignore_index=True,
                    )
                if hasattr(datasets, "test"):
                    sample_split_temp = dict(
                        sample_split,
                        **{
                            "SAMPLE_ID": datasets.test.sample_ids,
                            "SPLIT": ["test"] * len(datasets.test.sample_ids),
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
            if task == "RandomFeature":
                subtask = [
                    task + str(x) for x in range(1, 6)
                ]  
            for sub in subtask:
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
                                ml_type=ml_type
                            )
                    elif split_type.startswith("CV-"):
                        cv_folds = int(split_type.split("-")[1])

                        results = self._single_ml(
                                df=df,
                                clin_data=clin_data,
                                task_param=task_param,
                                sklearn_ml=sklearn_ml,
                                metric=metric,
                                cv_folds=cv_folds
                            )
                    elif split_type == "LOOCV":
                        # Leave One Out Cross Validation
                        results = self._single_ml(
                                df=df,
                                clin_data=clin_data,
                                task_param=task_param,
                                sklearn_ml=sklearn_ml,
                                metric=metric,
                                cv_folds=len(df)
                            )
                    else:
                        raise ValueError(
                            f"Your split type {split_type} is not supported. Please use 'use-split', 'CV-5', 'LOOCV' or 'CV-N'."
                        )


                    res_ml_alg = [str(sklearn_ml) for x in range(0, results.shape[0])]
                    res_ml_type = [ml_type for x in range(0, results.shape[0])]
                    res_ml_task = [task for x in range(0, results.shape[0])]
                    res_ml_subtask = [sub for x in range(0, results.shape[0])]

                    results["ML_ALG"] = res_ml_alg
                    results["ML_TYPE"] = res_ml_type
                    results["ML_TASK"] = res_ml_task
                    results["ML_SUBTASK"] = res_ml_subtask

                    df_results = pd.concat([df_results, results])


        result.embedding_evaluation = df_results

        return result
    
    @staticmethod
    def _single_ml(
            df,
            clin_data,
            task_param,
            sklearn_ml,
            metric,
            cv_folds=5,
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
                    metric_cv = [
                        metric for x in range(1, cv_folds + 1)
                    ]

                    score_df["cv_run"].extend(cv_runs)
                    score_df["score_split"].extend(split_cv)
                    score_df["CLINIC_PARAM"].extend(task_param_cv)
                    score_df["metric"].extend(metric_cv)
                    score_df["value"].extend(scores[m])

        return pd.DataFrame(score_df)

    @staticmethod
    def _single_ml_presplit(
        sample_split, df, clin_data, task_param, sklearn_ml, metric, ml_type
    ):
        """
        Function learns on the given data frame df and label data the provided sklearn model.
        Split infomation from autoencoder for samples are used and scores are returned as output as specified by metrics for each split
        ARGS:
            df (pd.DataFrame): Dataframe with input data
            clin_data (pd.DataFrame): Dataframe with label data
            task_param (str): Column name with label data
            sklearn_ml (sklearn.module): Sklearn ML module specifying the ML algorithm
            metrics (list): list of metrics (scores) to be calculated by cross validation
            cfg (dict): dictionary of the yaml config
        RETURNS:
            score_df (pd.DataFrame): data frame containing metrics (scores) for all CV runs (long format)

        """
        split_list = ["train", "valid", "test"]

        score_df = dict()
        score_df["score_split"] = list()
        score_df["CLINIC_PARAM"] = list()
        score_df["metric"] = list()
        score_df["value"] = list()

        X_train = df.loc[sample_split.loc[sample_split.SPLIT == "train", "SAMPLE_ID"], :]
        train_samples = [s for s in X_train.index]
        Y_train = clin_data.loc[train_samples, task_param]

        # train model once on training data
        if len(Y_train.unique()) > 1:
            sklearn_ml.fit(X_train, Y_train)

                # eval on all splits
            for split in split_list:
                X = df.loc[sample_split.loc[sample_split.SPLIT == split, "SAMPLE_ID"], :]
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
                        print(f"Classes in split {split} are not present in training data")
                        # Adjust Y to only contain classes present in Y_train
                        Y = Y[Y.isin(Y_train.unique())]
                        # Adjust X as well
                        X = X.loc[Y.index, :]
                    # y_proba = sklearn_ml.predict_proba(X)
                    # if len(pd.unique(Y)) == 2:
                    #     y_proba = y_proba[:, 1]

                score_temp = sklearn_scorer(sklearn_ml, X, Y)
                score_df["value"].append(score_temp)
                # match metric:
                #     case "roc_auc_ovo":
                #         # Check that Y has only classes which are present in Y_train
                #         if len(set(Y.unique()).difference(set(Y_train.unique()))) > 0:
                #             print(f"Classes in split {split} are not present in training data")
                #             # Adjust Y to only contain classes present in Y_train
                #             Y = Y[Y.isin(Y_train.unique())]
                #             # Adjust X as well
                #             X = X.loc[Y.index, :]
                #         y_proba = sklearn_ml.predict_proba(X)
                #         if len(pd.unique(Y)) == 2:
                #             y_proba = y_proba[:, 1]
                        
                #         roc_temp = roc_auc_score(
                #             Y, y_proba, multi_class="ovo", average="macro", labels=np.sort(Y_train.unique())
                #         )
                #         score_df["value"].append(roc_temp)

                #     case "r2":
                #         r2_temp = r2_score(Y, sklearn_ml.predict(X))
                #         score_df["value"].append(r2_temp)

                #     case _:
                #         logger.info(
                #             f"Your metric: {metric} is not yet supported or valid for this ML type"
                #         )

        return pd.DataFrame(score_df)
    
    @staticmethod
    def _get_ml_type(clin_data, task_param):
        ## Auto-Detection
        if type(list(clin_data[task_param])[0]) is str:
            ml_type = "classification"
        else:
            ml_type = "regression"

        return ml_type
    
    @staticmethod
    def _load_input_for_ml(task, dataset, result):

        final_epoch = result.model.config.epochs - 1

        if task == "Latent":
            df = pd.concat(
                [
                    result.get_latent_df(epoch=final_epoch, split="train"),
                    result.get_latent_df(epoch=final_epoch, split="valid"),
                    result.get_latent_df(epoch=-1, split="test"),
                ]
            )
        elif task in ["UMAP", "PCA","TSNE", "RandomFeature"]:
            df_processed = pd.concat([
                dataset.train._to_df(),
                dataset.test._to_df(),
                dataset.valid._to_df(),
            ])
            if task == "UMAP":                
                reducer = UMAP(n_components=result.model.config.latent_dim)
                df = pd.DataFrame(reducer.fit_transform(df_processed), index=df_processed.index)
            elif task == "PCA":
                reducer = PCA(n_components=result.model.config.latent_dim)
                df = pd.DataFrame(reducer.fit_transform(df_processed), index=df_processed.index)
            elif task == "TSNE":                
                reducer = TSNE(n_components=result.model.config.latent_dim)
                df = pd.DataFrame(reducer.fit_transform(df_processed), index=df_processed.index)
            elif task == "RandomFeature":
                df = df_processed.sample(
                    n=result.model.config.latent_dim, axis=1
                )
        else:
            raise ValueError(
                f"Your ML task {task} is not supported. Please use Latent, UMAP, PCA or RandomFeature."
            )
   
        return df

