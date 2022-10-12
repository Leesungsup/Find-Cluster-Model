# Find-Cluster-Model
## Objective
* Data analysis with below models:
    * K-Means, K-Medoids(CLARANS), DBSCAN, EM(Gaussian-Mixture), MeanShift
* Do various modeling with encoder, scalers, scoring, and metrics
    * Encoder : Label Encoder, Ordinal Encoder, OneHot Encoder
    * Scaler : Standard Scaler, MinMax Scaler, Robust Scaler, MinAbs Scaler
    * Scoring Method : Silhouette Index, Elbow method, Purity
    * Metrics : Euclidean, Manhattan
* Do comparisons in various ways
    * With parameters : Hyperparameter-tuning (Encoder, Scaler, Scoring, Metric,...)
    * A cluster within our hands (hand-calculated from the dropped **median_house_value**)
* Visualization (Plot) of clustering results
    * Print the result (score)
    * Scatter plot of cluster results
    * Plot of result scores
* Performance checks (Comparison)
    * Do the timer (timeout) for data analysis, check how much time the analysis takes
    * With timeout, skip the time-taking modeling method and modify it.

## Tuning Model - Basic class for hyperparameter tuning
* Parameters 
    * param: dictionary, default=None
        * parameters for hyperparameter tuning – The TuningModel class saves it and sends them to model as keyword parameters when the tuning model is in fit().
    * timer: integer, default=900
        * Attribute for timeout feature (second), data analysis of each model cannot exceed it.
* Attributes
    * Params: dictionary
        * parameter input from constructor
    * model_list: list(Any)
        * model list, available after using fit()
    * best_params_: dictionary
        * best model's parameters, available after using fit()
    * best_score_: float
        * best model's score, available after using fit()
    * best_model_: Any
        * best model, available after using fit()
    * _dataset: numpy.ndarray 
        * A dataframe values used in fit()
    * __timer: integer
        * Attribute for timeout feature, in seconds.

> _model_execute(x,params,result)

Do data analysis (modeling). Since this class is a parent class, there are no specific contexts in here.
* Parameters 
    * x: numpy.ndarray
        * Dataframe values for clustering
    * params: dictionary
        * Chosen parameter from whole parameters. This method is not specified in TuningModel – you need to specify in its child classes.
    * return_dict: multiprocessing.Manager.dict
        * Shared variable from multiprocessing.Manager() used to send the result of analysis (model)
* Returns
    * None (use return_dict)

> _model_scoring(model, x)

Do scoring. Since this class is a parent class, there are no specific contexts in here.
* Parameter
    * model: numpy.ndarray
        * A model obtained from _model_execute().
    * x: dictionary
        * Dataframe values used for data analysis – it is used for scoring.
* Returns
    * score: float
        * The calculated score

    > _is_best(model)
    
In scoring, this function decides the better data models with its result – (ex; higher score of silhouette index, lower score of Elbow methods, …) Since most of children uses silhouette index, this function is defined in higher score priority.
* Parameters
    * model : dictionary
        * A dictionary containing model, param, scores
* Returns
    * result : boolean
        * True or False

> check_best()

Compares the scores in model list. Since the child classes can have difference in scoring, we declared the scoring-act function in separated.
* Parameters
    * None
* Returns
    * None

> fit(x)

Do data analysis for whole parameters (= create the submodels using hyperparameter-tuning)
* Parameters
    * x : numpy.ndarray
        * A numpy array for data analysis, values of dataframe
* Returns
    * None 
        * The results are saved in self.model_list and self.best_**params.

> plot_score(title)

Plot the score-parameter relationship. This function calculate the score of specific param and average of other parameters and plot it on subplot.
* Parameters
    * title: str, default=None
        * Title of subplots – you can declare it on matplotlib.pyplot.suptitle.
* Returns
    * None
        * Since this function uses the pyplot, you should use pyplot.show() function to see results.
    
* Child class of **Tuning Model**
    * ClaransTune
    * DbscanTune
    * KMeansTune
    * GMMTune

* Class for hyperparameter tuning
    * ClaransTune_Purity
    * DbscanTune_Purity
    * KMeansTune_Purity
    * GMMTune_Purity

* Class for elbow method
    * KMeansTune_Elbow

## major_function
An **AutoML** function for various encoder, scaler, cluster, and distance methods. This function makes:
> len(encoder) * len(scaler) * (len(cluster[i]) * len(param[cluster[i]][0]) * len(param[cluster[i]][1]) * …) * …

…data analysis. Due to the performance issue (too much time taken), we put the timeout feature on tuning class.
* Parameters
    * x : pandas.DataFrame
        * A data frame for data analysis. It is converted into numpy.ndarray when clustering.
    * encoder : set
        * Encoders to use. Make sure using appropriate encoders to avoid the data type errors.
    * scaler : set
        * Scalers to use
    * cluster : dict (type(cluster), dict(params...)
        * Cluster models for data analysis. Make sure you use the supported cluster model 
        ```python   
        supported_model = { 
                KMeans: KMeansTune,
                clarans: ClaransTune,
                DBSCAN: DbscanTune,
                GaussianMixture: GMMTune,
                MeanShift: MeanShiftTune
        }
You can write cluster types as key, and dictionary contains the parameters of the cluster. At least one parameter input required.

* Returns
    * output: list(dict)
        * Output hyperparameter tuning models containing encoder, scaler, dataframe, and model.
            ```python
            supported_model = {
                'encoder': encoder,
                'scaler': scaler,
                'model': model,
                'dataframe': dataframe
            }
            ```

Example usage)
```python
encoder_list = {LabelEncoder, OrdinalEncoder}
scalar_list = {StandardScaler, MinMaxScaler}
cluster_list = {
    KMeans: {
        'n_clusters': [2, 3, 4],
        'init': ['k-means++', 'random']
    },
    clarans: {
        'number_clusters': [2, 3, 4],
        'numlocal': [2, 4, 6],
        'maxneighbor': [3, 5, 7]
    },
    DBSCAN: {
        'eps': [0.01, 0.05, 0.1],
        'min_samples': range(2, 6)
    },
    GaussianMixture: {
        'n_components': [2, 3, 4]
    },
    MeanShift:{
     'bandwidth' : [0.8,1.6,3.0]
    }
}
method_list = {'euclidean', 'manhattan'}

```
