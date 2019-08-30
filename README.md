Sales_forecasting
==============================

The project objective is to show the Prophet libraries capabilities on simulated sales data.

-  Simulated sales data:

        The sales will refer to different products for different clients
        (multiple products can be attached to only one client : many to many). 
        
        Every product will be attached to a cluster of products. 
        A cluster of products is attached to a specific machine settings. 
        
        This will provide a tool for the production optimization. 
        If we can group the products while using the forecasting tools,
        that could give an edge to the company we deliver this tool to.
        
The data has been simulated included 3 days random noise in the dates and 10% random noise in the quantities.

## Two tables will be generated:
- The first table will attach every product to a cluster of products (5 clusters A, B, C, D and E will be used):

| Product_code  |  Associated cluster |
|---|---|
| CLA01  |  A  |
| CLA02  |   E  |
| CLB01  |   A  |

The product code 3 first characters define the client : CLA : Client A , CLB : Client B etc.

- The second table is the actual Sales history table:

| Product_code  | Date  |  Quantity |
|---|---|---|
|  CLB01 | 25/07/2019  | 1,000  |
|  CLB01 | 19/07/2019  |  1,500 |
|   CLA02 | 23/07/2019  |  10,000 |


The Predicted sales appear in the graph below:

![Image of Predicted sales](https://github.com/raphaelribard/Sales_forecasting/blob/master/docs/Predicted_sales.png)


The predictions performance is assessed using the following joint plot: 

- The vertical axis shows the errors in the quantities values (in %)
- The horizontal axis shows the error in the number of days for the predicted dates (in days)
<img src="https://github.com/raphaelribard/Sales_forecasting/blob/master/docs/Predictions_performance.png" width="500">

We can see that the algorithm captured the simulated noise (3 days random noise and 10% random noise for the quantities).

The outputs for the client are the following ones : (one full calendar with all the predicted sales and one calendar just for the following week (one clean version and one with the product level predicted and actual sales)):

Full calendar:

![Image of Full calendar](https://github.com/raphaelribard/Sales_forecasting/blob/development/docs/OverallCalendar.png)

Clean calendar for the following week:

![Image of Clean calendar](https://github.com/raphaelribard/Sales_forecasting/blob/development/docs/CleanWeeklyCalendar.png)

Calendar for the following week with product-level sales:

![Image of Weekly calendar](https://github.com/raphaelribard/Sales_forecasting/blob/development/docs/WeeklyCalendarPlusProductLevel.png)


# TO-DO List:
-  Redo it at a cluster level instead of a product level
-  Try to incorporate seasonality impact in the simulated dataset
-  Try the R changepoint package from Lancaster University
-  Fully use the CookieCutter structure
-  Integrate it in AWS

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
