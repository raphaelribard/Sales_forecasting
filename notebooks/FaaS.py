import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm

from IPython.core.display import HTML

from fbprophet import Prophet
from fbprophet.plot import plot_plotly

import plotly.offline as py
import plotly.graph_objs as go
import plotly.express as px


class SalesForecaster:
    """This class creates 'easy to handle' forecaster objects
    It will gather all the required variables to make the code more readable

    -  sales_clusters_df (pandas dataframe): The original sales dataframe
        The columns are :
            - product_code : string values such as CLA0 (CLA is the client and 0 is the product number)
            - date : datetime64 (ns) the date of the sale such as pd.to_datetime("2018-01-02") : YYYY-MM-DD
            - quantity : int64 an integer value: the number of products for this sale
            - cluster : int64 an integer value The cluster the product is part of
    -  test_date (string : "2019-03-01" : YYYY-MM-DD): the training data is automatically all sales prior to this date
    -  max_waiting_time (string such as '7 days') : The maximum time a client is willing to wait :
                                                    required for grouping orders into batches)
    -  calendar_length (string such as '7 days'): The calendar length you want to zoom in

    """

    def __init__(self,
                 sales_clusters_df,
                 test_date,
                 max_waiting_time,
                 detailed_view=False,
                 calendar_length='7 days'
                 ):

        self.sales_clusters_df = sales_clusters_df
        self.test_date = test_date
        self.max_waiting_time = max_waiting_time
        self.detailed_view = detailed_view
        self.calendar_length = calendar_length

    def get_predicted_batches(self):

        """This function takes the original sales df,
        computes the dates and quantities models at a product level using the test_date to split the dataset
        into a training dataset and a testing dataset,
        generates the predicted sales,
        computes the associated "predicted" batches using the max waiting time value,
        computes the optimal batches using the actual data using the max waiting time value,
        outputs the optimal batches df and the predicted batches df,
        and 2 graphs to visualize it:

        -  Input:
        All the inputs are encapsulated in the SalesForecaster instance:
            -  sales_clusters_df
            -  test_date
            -  max_waiting_time
            -  calendar_length

        -  Output:
          - Main graph with optimal batches vs predicted batches for the test data
          - The same graph zoomed in the week following the test date
          - 1 optimal batches df
          - 1 predicted batches df
        """

        clusters_list = self.sales_clusters_df['Cluster'].unique()

        optimal_batches = []
        predicted_batches = []
        predictions = []

        for cluster in clusters_list:
            local_optimal_batches, local_predicted_batches, local_predictions = self.\
                get_cluster_level_predicted_batches(cluster)
            local_optimal_batches['Cluster'] = cluster
            local_predicted_batches['Cluster'] = cluster
            optimal_batches.append(local_optimal_batches)
            predicted_batches.append(local_predicted_batches)
            predictions.append(local_predictions)

        optimal_batches = pd.concat(optimal_batches)
        optimal_batches.reset_index(drop=True,
                                    inplace=True)
        optimal_batches['batch_date'] = optimal_batches.batch_date.str.split(' ').apply(lambda x: x[0])

        predicted_batches = pd.concat(predicted_batches)
        predicted_batches.reset_index(drop=True,
                                      inplace=True)
        predicted_batches['batch_date'] = predicted_batches.batch_date.str.split(' ').apply(lambda x: x[0])

        predictions = pd.concat(predictions)
        predictions.reset_index(drop=True,
                                inplace=True)

        dark_map = px.colors.qualitative.Dark2
        pastel_map = px.colors.qualitative.Pastel2

        fig = go.Figure()

        for (cluster, dark_color, pastel_color) in zip(clusters_list, dark_map, pastel_map):
            local_optimal = optimal_batches[optimal_batches['Cluster'] == cluster]
            local_predicted = predicted_batches[predicted_batches['Cluster'] == cluster]
            fig.add_trace(go.Bar(x=pd.to_datetime(local_optimal[local_optimal['batch_date'] > self.test_date] \
                                                      ['batch_date']) - pd.Timedelta('12 hours'),
                                 y=local_optimal[local_optimal['batch_date'] > self.test_date] \
                                     ['quantities'],
                                 name='Cluster #{}\nOptimized batches - actual values'.format(cluster),
                                 width=1e3 * pd.Timedelta('6 hours').total_seconds(),
                                 marker_color=dark_color))

            fig.add_trace(go.Bar(x=pd.to_datetime(local_predicted[local_predicted['batch_date'] > self.test_date] \
                                                      ['batch_date']) - pd.Timedelta('12 hours'),
                                 y=local_predicted[local_predicted['batch_date'] > self.test_date] \
                                     ['predicted_quantities'],
                                 name='Cluster #{}\nPredicted batches'.format(cluster),
                                 width=1e3 * pd.Timedelta('6 hours').total_seconds(),
                                 marker_color=pastel_color))

        # Edit the layout
        fig.update_layout(title='Optimal batches vs predicted batches for the test period',
                          xaxis_title='Date',
                          yaxis_title='Quantities')
        fig.show()

        fig = go.Figure()

        for (cluster, dark_color, pastel_color) in zip(clusters_list, dark_map, pastel_map):
            local_optimal = optimal_batches[optimal_batches['Cluster'] == cluster]
            local_predicted = predicted_batches[predicted_batches['Cluster'] == cluster]

            fig.add_trace(go.Bar(x=pd.to_datetime(local_optimal[(local_optimal['batch_date'] > self.test_date) & \
                                                                (local_optimal['batch_date'] < str((pd.Timestamp(
                                                                    self.test_date) + pd.Timedelta(self.calendar_length))))] \
                                                      ['batch_date']) - pd.Timedelta('0 hours'),
                                 y=local_optimal[(local_optimal['batch_date'] > self.test_date) & \
                                                 (local_optimal['batch_date'] < str(
                                                     (pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))] \
                                     ['quantities'],
                                 name='Cluster #{}\nOptimized batches - actual values'.format(cluster),
                                 width=1e3 * pd.Timedelta('6 hours').total_seconds(),
                                 marker_color=dark_color,
                                 marker_line_color='black',
                                 marker_line_width=1.5,
                                 opacity=0.6))

            fig.add_trace(go.Bar(x=pd.to_datetime(local_predicted[(local_predicted['batch_date'] > self.test_date) & \
                                                                  (local_predicted['batch_date'] < str((pd.Timestamp(
                                                                      self.test_date) + pd.Timedelta(self.calendar_length))))] \
                                                      ['batch_date']) - pd.Timedelta('0 hours'),
                                 y=local_predicted[(local_predicted['batch_date'] > self.test_date) & \
                                                   (local_predicted['batch_date'] < str(
                                                       (pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))] \
                                     ['predicted_quantities'],
                                 name='Cluster #{}\nPredicted batches'.format(cluster),
                                 width=1e3 * pd.Timedelta('6 hours').total_seconds(),
                                 marker_color=pastel_color,
                                 marker_line_color='black',
                                 marker_line_width=1.5,
                                 opacity=0.6))

        # Edit the layout
        fig.update_layout(barmode='stack', xaxis_tickangle=-45,
                          title='Optimal batches vs predicted batches for the following week',
                          xaxis_title='Date',
                          yaxis_title='Quantities')
        fig.show()

        fig = go.Figure()

        for (cluster, dark_color, pastel_color) in zip(clusters_list, dark_map, pastel_map):
            local_optimal = optimal_batches[optimal_batches['Cluster'] == cluster]
            local_predicted = predicted_batches[predicted_batches['Cluster'] == cluster]
            local_predictions = predictions[predictions['Cluster'] == cluster]

            if local_predictions[(local_predictions.ds > self.test_date) & (
                    local_predictions.ds <= str((pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))].shape[
                0] > 0:
                display(HTML(local_predictions[(local_predictions.ds > self.test_date) & (
                        local_predictions.ds <= str((pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))][
                                 ['ds', 'y', 'product_code', 'Cluster']].to_html()))

            if local_predictions[(local_predictions.yhat_date > self.test_date) & (
                    local_predictions.yhat_date <= str(
                (pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))].shape[
                0] > 0:
                display(HTML(local_predictions[(local_predictions.yhat_date > self.test_date) & (
                        local_predictions.yhat_date <= str((pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))][
                                 ['yhat_date', 'yhat_qty', 'product_code', 'Cluster']].to_html()))

            fig.add_trace(go.Bar(x=pd.to_datetime(local_optimal[(local_optimal['batch_date'] > self.test_date) & \
                                                                (local_optimal['batch_date'] <= str((pd.Timestamp(
                                                                    self.test_date) + pd.Timedelta(self.calendar_length))))] \
                                                      ['batch_date']) - pd.Timedelta('0 hours'),
                                 y=local_optimal[(local_optimal['batch_date'] > self.test_date) & \
                                                 (local_optimal['batch_date'] <= str(
                                                     (pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))] \
                                     ['quantities'],
                                 name='Cluster #{}\nOptimized batches - actual values'.format(cluster),
                                 width=1e3 * pd.Timedelta('6 hours').total_seconds(),
                                 marker_color=dark_color,
                                 marker_line_color='black',
                                 marker_line_width=1.5,
                                 opacity=0.6))

            fig.add_trace(go.Bar(x=pd.to_datetime(local_predicted[(local_predicted['batch_date'] > self.test_date) & \
                                                                  (local_predicted['batch_date'] <= str((pd.Timestamp(
                                                                      self.test_date) + pd.Timedelta(self.calendar_length))))] \
                                                      ['batch_date']) - pd.Timedelta('0 hours'),
                                 y=local_predicted[(local_predicted['batch_date'] > self.test_date) & \
                                                   (local_predicted['batch_date'] <= str(
                                                       (pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))] \
                                     ['predicted_quantities'],
                                 name='Cluster #{}\nPredicted batches'.format(cluster),
                                 width=1e3 * pd.Timedelta('6 hours').total_seconds(),
                                 marker_color=pastel_color,
                                 marker_line_color='black',
                                 marker_line_width=1.5,
                                 opacity=0.6))

            fig.add_trace(go.Scatter(x=pd.to_datetime(local_predictions[(local_predictions.ds > self.test_date) & (
                    local_predictions.ds <= str((pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))]['ds']),
                                     y=local_predictions[
                                         (local_predictions.ds > self.test_date) & (local_predictions.ds <= str(
                                             (pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))]['y'],
                                     marker=dict(
                                         color=dark_color,
                                         size=10,
                                         line=dict(
                                             color='white',
                                             width=2
                                         )
                                     ),
                                     mode='markers',
                                     name='actual_sales'))

            fig.add_trace(go.Scatter(x=pd.to_datetime(local_predictions[(local_predictions.yhat_date > self.test_date) & (
                    local_predictions.yhat_date <= str((pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))][
                                                          'yhat_date']),
                                     y=local_predictions[(local_predictions.yhat_date > self.test_date) & (
                                             local_predictions.yhat_date <= str(
                                         (pd.Timestamp(self.test_date) + pd.Timedelta(self.calendar_length))))]['yhat_qty'],
                                     marker=dict(
                                         color=pastel_color,
                                         size=10,
                                         line=dict(
                                             color='white',
                                             width=2
                                         )
                                     ),
                                     mode='markers',
                                     name='predicted_sales'))

        # Edit the layout
        fig.update_layout(barmode='stack', xaxis_tickangle=-45,
                          title='Optimal batches vs predicted batches for the following week \nPLUS product_code level sales (predicted and actual)',
                          xaxis_title='Date',
                          yaxis_title='Quantities')
        fig.show()

        local_predictions = predictions[predictions['ds'] > self.test_date]

        sns.set(style="white")
        # Show the joint distribution using kernel density estimation
        g = sns.jointplot(
            pd.Series(local_predictions['error_days'].values / (24 * 60 * 60 * 1e9), name='error_days\nin days'),
            pd.Series(local_predictions['error_quantities'].values, name='error_quantities\nin%'),
            kind="kde", height=7, space=0)

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=predictions['ds'],
                                 y=predictions['y'],
                                 marker=dict(
                                     color='LightSkyBlue',
                                     size=10,
                                     line=dict(
                                         color='white',
                                         width=2
                                     )
                                 ),
                                 mode='markers',
                                 name='actual_y'))

        fig.add_trace(go.Scatter(x=local_predictions['yhat_date'],
                                 y=local_predictions['yhat_qty'].astype(int),
                                 opacity=0.5,
                                 marker=dict(
                                     size=7,
                                     line=dict(
                                         color='MediumPurple',
                                         width=2
                                     )
                                 ),
                                 mode='markers',
                                 name='yhat'))

        fig.update_layout(
            shapes=[
                # Line Vertical
                go.layout.Shape(
                    type="line",
                    x0=self.test_date,
                    y0=np.min(predictions['y']),
                    x1=self.test_date,
                    y1=np.max(predictions['y']),
                    line=dict(
                        color="grey",
                        width=2,
                        dash="dashdot"
                    )
                )
            ]
        )

        fig.add_trace(go.Scatter(
            x=[pd.Timestamp(self.test_date) - pd.Timedelta('90 days'),
               pd.Timestamp(self.test_date) + pd.Timedelta('90 days')],
            y=[np.min(predictions['y']),
               np.min(predictions['y'])],
            text=["Training data <--",
                  "--> Testing data"],
            mode="text",
            name='training and test data'))

        # Edit the layout
        fig.update_layout(title='All predicted dates and quantities',
                          xaxis_title='Date',
                          yaxis_title='Sales quantities')

        fig.show()

        return optimal_batches, predicted_batches, predictions

    # Optimization part

    def print_msg(self,
                  msg):

        """This function is just a function to highlight a text while printing it"""
        length = len(msg)
        print(length * '-' + '\n' + msg + '\n' + length * '-')

        return

    def optimize_orders_processing(self,
                                   sales_df,
                                   cluster):
        sales_cluster = sales_df[sales_df['Cluster'] == cluster][
            ['date', 'quantity', 'product_code']]. \
            sort_values(by='date')

        sales_cluster = sales_cluster.reset_index(drop=True)

        maximum_wait_time = self.max_waiting_time.split(' ')[0] + self.max_waiting_time.split(' ')[1][0]

        sales_cluster = sales_cluster.set_index('date')

        sales_cluster['number_of_previous_neighbours'] = sales_cluster[['quantity']].rolling(
            maximum_wait_time).count()

        sales_cluster['batch'] = 0
        sales_cluster['batch_date'] = sales_cluster.index.values

        i = 0
        batch_group = 1

        for i in tqdm(range(sales_cluster.shape[0])):

            win_edge = sales_cluster.sort_values('number_of_previous_neighbours', ascending=False). \
                index[i]

            if (sales_cluster.batch[(sales_cluster.index <= win_edge)
                                    & (sales_cluster.index > (
                    win_edge - pd.Timedelta(self.max_waiting_time)))].sum()) == 0:
                #         print(i)
                sales_cluster.batch.where(~((sales_cluster.index <= win_edge)
                                            & (sales_cluster.index > (win_edge - pd.Timedelta(self.max_waiting_time)))),
                                          batch_group,
                                          inplace=True)

                win = sales_cluster.batch_date[((sales_cluster.index <= win_edge)
                                                & (sales_cluster.index > (
                                win_edge - pd.Timedelta(self.max_waiting_time))))]

                batch_date = win_edge - ((win.iloc[-1] - win.iloc[0]) / 2)

                sales_cluster.batch_date[((sales_cluster.index <= win_edge)
                                          & (sales_cluster.index >
                                             (win_edge - pd.Timedelta(self.max_waiting_time))))] = batch_date

                batch_group = batch_group + 1

            i = i + 1

        reduce_workload = 100 * (1 - (sales_cluster.batch.value_counts().shape[0] / sales_cluster.shape[0]))

        self.print_msg('The workload is reduced by {}%'.format(str(int(reduce_workload))))

        batch_dates_df = sales_cluster[['batch', 'batch_date']].reset_index(drop=True).drop_duplicates()

        batch_dates_df['batch'] = np.max(
            self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['quantity'].values)

        if self.detailed_view:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['date'].astype(str).values,
                    y=self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['quantity'].values,
                    mode='markers',
                    name='Cluster ' + str(cluster)))

            fig.add_trace(go.Bar(x=batch_dates_df['batch_date'], y=batch_dates_df['batch'],
                                 marker_color='lightgrey',
                                 name='Batch dates'))

            for batch in sales_cluster.batch.value_counts().index.values:
                fig.add_trace(go.Scatter(x=sales_cluster[sales_cluster['batch'] == batch].index.astype(str).values,
                                         y=sales_cluster[sales_cluster['batch'] == batch]['quantity'].values,
                                         mode='markers',
                                         name='batch ' + str(batch)))

            # Edit the layout
            fig.update_layout(title='Optimized batches for cluster#' + str(cluster),
                              xaxis_title='Date',
                              yaxis_title='Quantities')

            fig.update_yaxes(
                range=[np.min(self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['quantity'].values),
                       np.max(self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['quantity'].values)])

            fig.show()

        return sales_cluster

    # forecast_part

    def generate_models_n_predictions_df_n_plotly_viz(self,
                                                      product_code):
        """ This function aims to take a product_code and output:
            -  1 date model
            -  1 qty model
            -  1 predictions df
            -  1 plotly graph of the predictions
        """
        test_year = self.test_date.split('-')[0]
        # First select the product_code df with the sales dates and copy it into a new df
        prophet_date_df = self.sales_clusters_df[self.sales_clusters_df['product_code'] == product_code][
            ['date']].copy()
        prophet_date_df.reset_index(drop=True, inplace=True)

        # Using global variable "test_year", generate the index where we split the df into a train year and a test year
        idx_split = prophet_date_df['date'][(prophet_date_df['date'] >= test_year).cumsum() == 1].index[0]

        # Generate the next sale date column which will be necessary to generate the "number of days to next sale" feature
        # This is the feature the "date model" will have to predict
        prophet_date_df = self.generate_next_sale_column(prophet_date_df)

        # Refactor the column names for the Facebook Prophet library
        prophet_date_df.rename(index=str, columns={"date": "ds"},
                               inplace=True)

        # Create the "date model"
        next_sale_date_model = Prophet()

        # Train the model on the training year
        if self.detailed_view:
            self.print_msg('--> Now training the "date model"')
        next_sale_date_model.fit(prophet_date_df[:idx_split - 1])

        # Make predictions on the complete df (training + test year) : this will let us check the overfitting later on
        next_sale_date_forecast = next_sale_date_model.predict(pd.DataFrame(prophet_date_df['ds']))

        # This is if you want to plot the date plot
        # plot_date_predictions(next_sale_date_forecast,product_code,idx_split)

        # Select the product_code df with the sales dates AND quantities and copy it into a new df
        prophet_qty_df = self.sales_clusters_df[self.sales_clusters_df['product_code'] == product_code][
            ['date', 'quantity']].copy()

        # Make it prophet compliant
        prophet_qty_df.rename(index=str,
                              columns={"date": "ds",
                                       "quantity": "y"},
                              inplace=True)

        # Instantiate the quantity model
        qty_model = Prophet()

        if self.detailed_view:
            self.print_msg('--> Now training the "quantity model"')

        qty_model.fit(prophet_qty_df[:idx_split])

        # Make predictions on the complete df (training + test year) : this will let us check the overfitting later on
        qty_forecast = qty_model.predict(pd.DataFrame(prophet_qty_df['ds']))

        # Select the interesting columns for the qty df
        qty_predictions = qty_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Renaming the prophet output columns to identify the quantity predictions (in comparison with the sales dates predictions)
        for col in qty_predictions.columns[1:]:
            qty_predictions.rename(index=str,
                                   columns={col: col + '_qty'},
                                   inplace=True)

        # Select the interesting columns for the sales df
        next_sale_dates_predictions = next_sale_date_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Make the sales df readable and easy to identify
        for col in next_sale_dates_predictions.columns[1:]:
            next_sale_dates_predictions[col] = [pd.Timedelta(str(int(t)) + ' days') for t in
                                                next_sale_dates_predictions[col].values]
            next_sale_dates_predictions[col] = next_sale_dates_predictions['ds'] + next_sale_dates_predictions[col]
            next_sale_dates_predictions.rename(index=str,
                                               columns={col: col + '_date'},
                                               inplace=True)

        # Prepare the full_predictions df which will include dates, quuantities predictions and the actual values
        # (dates and quantities)
        full_predictions_df = qty_predictions.copy()
        # Sales dates and quantities concatenation
        for col in next_sale_dates_predictions.columns[1:]:
            local_list = next_sale_dates_predictions[col].values.tolist()
            local_list.insert(0, np.nan)
            full_predictions_df[col] = pd.to_datetime(local_list)

        # Include the actual values in the full_predictions_df
        full_predictions_df['y'] = prophet_qty_df['y'].values

        if self.detailed_view:
            # Plot it (fancy with plotly)
            self.plot_full_predictions(full_predictions_df,
                                       product_code,
                                       idx_split)

        return (next_sale_date_model,
                qty_model,
                full_predictions_df)

    def generate_next_sale_column(self,
                                  df):
        df['next_sale_date'] = df['date'].shift(-1)
        df.drop([len(df) - 1], inplace=True)

        # y will be the number of days until the next sale
        df['y'] = (df['next_sale_date'] - df['date']).dt.days.astype(int)

        return df

    def plot_full_predictions(self,
                              full_predictions_df, product_code, idx_split):
        fig = go.Figure()

        fig.add_trace(go.Scatter(x=full_predictions_df['ds'],
                                 y=full_predictions_df['y'],
                                 marker=dict(
                                     color='LightSkyBlue',
                                     size=20,
                                     line=dict(
                                         color='black',
                                         width=2
                                     )
                                 ),
                                 mode='markers',
                                 name='actual_y'))

        fig.add_trace(go.Scatter(x=full_predictions_df['yhat_date'][idx_split:],
                                 y=full_predictions_df['yhat_qty'][idx_split:].astype(int),
                                 opacity=0.5,
                                 marker=dict(
                                     size=15,
                                     line=dict(
                                         color='MediumPurple',
                                         width=2
                                     )
                                 ),
                                 mode='markers',
                                 name='yhat'))

        fig.add_trace(go.Scatter(x=full_predictions_df['yhat_lower_date'][idx_split:],
                                 y=full_predictions_df['yhat_lower_qty'][idx_split:].astype(int),
                                 opacity=0.5,
                                 marker=dict(
                                     size=10,
                                     line=dict(
                                         color='MediumPurple',
                                         width=2
                                     )
                                 ),
                                 mode='markers',
                                 name='yhat_lower'))

        fig.add_trace(go.Scatter(x=full_predictions_df['yhat_upper_date'][idx_split:],
                                 y=full_predictions_df['yhat_upper_qty'][idx_split:].astype(int),
                                 opacity=0.5,
                                 marker=dict(
                                     size=10,
                                     line=dict(
                                         color='MediumPurple',
                                         width=2
                                     )
                                 ),
                                 mode='markers',
                                 name='yhat_upper'))

        fig.update_layout(
            shapes=[
                # Line Vertical
                go.layout.Shape(
                    type="line",
                    x0=full_predictions_df['ds'][idx_split - 1],
                    y0=np.min(full_predictions_df['y']),
                    x1=full_predictions_df['ds'][idx_split - 1],
                    y1=np.max(full_predictions_df['y']),
                    line=dict(
                        color="grey",
                        width=2,
                        dash="dashdot"
                    )
                )
            ]
        )

        fig.add_trace(go.Scatter(
            x=[full_predictions_df['ds'][idx_split - 2],
               full_predictions_df['ds'][idx_split]],
            y=[np.mean(full_predictions_df['y']),
               np.mean(full_predictions_df['y'])],
            text=["Training data <--",
                  "--> Testing data"],
            mode="text",
            name='training and test data'
        ))

        # Edit the layout
        fig.update_layout(title='Predicted dates and quantities for ' + product_code,
                          xaxis_title='Date',
                          yaxis_title='Sales quantities')

        fig.show()

        return

    def get_aggregated_predictions(self,
                                   cluster):
        cluster_product_codes = self.sales_clusters_df[self.sales_clusters_df.Cluster == cluster]['product_code']. \
            unique()

        cluster_predictions = []
        for prod in cluster_product_codes:
            cluster_predictions.append(
                self.generate_models_n_predictions_df_n_plotly_viz(prod))

        cluster_aggregated_predictions = cluster_predictions[0][2]
        cluster_aggregated_predictions['product_code'] = cluster_product_codes[0]
        for i, prod in enumerate(cluster_product_codes[1:]):
            #         print(i)
            #         print(prod)
            local_df = cluster_predictions[i + 1][2]
            local_df['product_code'] = prod
            cluster_aggregated_predictions = pd.concat([cluster_aggregated_predictions, local_df])

        cluster_predictions_test_year = cluster_aggregated_predictions[cluster_aggregated_predictions['yhat_date'] >
                                                                       self.test_date]

        if self.detailed_view:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['date'].astype(str).values,
                    y=self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['quantity'].values,
                    mode='markers',
                    name='Cluster ' + str(cluster)))

            fig.add_trace(go.Scatter(x=cluster_predictions_test_year['yhat_date'].astype(str).values,
                                     y=cluster_predictions_test_year['yhat_qty'].values,
                                     mode='markers',
                                     name='Cluster ' + str(cluster) + ' predictions'))

            fig.update_layout(
                shapes=[
                    # Line Vertical
                    go.layout.Shape(
                        type="line",
                        x0=self.test_date,
                        y0=np.min(
                            self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['quantity'].values),
                        x1=self.test_date,
                        y1=np.max(
                            self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['quantity'].values),
                        line=dict(
                            color="grey",
                            width=2,
                            dash="dashdot"
                        )
                    )
                ]
            )

            fig.add_trace(go.Scatter(
                x=[str(int(self.test_date.split('-')[0]) - 1) + '-10-10',
                   self.test_date.split('-')[0] + '-03-10'],
                y=[np.min(self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['quantity'].values),
                   np.min(self.sales_clusters_df[self.sales_clusters_df['Cluster'] == cluster]['quantity'].values)],
                text=["Training data <--",
                      "--> Testing data"],
                mode="text",
                name='training and test data'
            ))

            # Edit the layout
            fig.update_layout(title='Labels Sales for cluster #' + str(cluster),
                              xaxis_title='Date',
                              yaxis_title='Quantities')

            fig.show()

        cluster_aggregated_predictions['error_days'] = cluster_aggregated_predictions['yhat_date'] - \
                                                       cluster_aggregated_predictions['ds']

        cluster_aggregated_predictions['error_quantities'] = 100 * (cluster_aggregated_predictions['yhat_qty'] -
                                                                    cluster_aggregated_predictions['y']) \
                                                             / cluster_aggregated_predictions['y']

        return cluster_aggregated_predictions

    def get_cluster_level_predicted_batches(self,
                                            cluster):
        '''
        This function is working at a cluster level: it takes a cluster and
        predicts the batches for the test year, then it compares it with the actual data

        - Inputs:
            -  The original sales dataframe generated by the simulated data generator notebook,
            -  A cluster (a group of product_codes with similar machine settings),
            -  A maximum waiting time (for optimization and batches generation),
            -  A test_date (to know what dat should be used to train the forecast models)
        -  Outputs:
            - The graph comparing the best optimal batches (with the actual data if it is available) and the predicted batches with the predicted quantities over the actual quantities delta
            - The 2 dataframes :
                -  1 df with the optimal batches (dates, quantities, associated product_codes and dates)
                -  1 df with the predicted batches (predicted dates, predicted quantities, associated product_codes and associated dates)
            - The dates and quantities models if required (WIP)
        '''

        self.print_msg('Optimizing the actual data for cluster #{}'.format(cluster))

        sales_cluster_optimization = self.optimize_orders_processing(self.sales_clusters_df, cluster)

        self.print_msg('Creating and aggregating the Prophet/Sales forecasting models for cluster #{}'.format(cluster))

        cluster_aggregated_predictions = self.get_aggregated_predictions(cluster)

        full_predictions = cluster_aggregated_predictions.copy()
        full_predictions['Cluster'] = cluster

        cluster_aggregated_predictions = cluster_aggregated_predictions.\
            rename(index=str,
                    columns={"yhat_date": "date",
                             "yhat_qty": "quantity"})

        cluster_aggregated_predictions = cluster_aggregated_predictions[
            ~pd.isnull(cluster_aggregated_predictions['date'])]

        cluster_aggregated_predictions['Cluster'] = cluster

        self.print_msg('Optimizing the predicted data')

        sales_cluster_optimized_predictions = self.optimize_orders_processing(cluster_aggregated_predictions, cluster)

        batch_dates_n_predicted_quantities_cluster = self.transform_the_optimized_df(sales_cluster_optimized_predictions,
                                                                                "predicted_quantities")
        batch_dates_n_quantities_cluster = self.transform_the_optimized_df(sales_cluster_optimization, 'quantities')

        batch_dates_n_quantities_cluster.reset_index(inplace=True)

        batch_dates_n_predicted_quantities_cluster.reset_index(inplace=True)

        test_year = self.test_date.split('-')[0]

        total_predicted_quantities = np.sum(batch_dates_n_predicted_quantities_cluster['predicted_quantities'])

        total_quantities = np.sum(batch_dates_n_quantities_cluster['quantities'])

        predictions_coverage = 100 * (total_predicted_quantities / total_quantities)

        self.print_msg('The predicted quantities represent {:.1f}% of the actual quantities'.format(predictions_coverage))

        if self.detailed_view:

            fig = go.Figure()

            fig.add_trace(
                go.Bar(x=batch_dates_n_quantities_cluster[batch_dates_n_quantities_cluster['batch_date'] > test_year] \
                    ['batch_date'],
                       y=batch_dates_n_quantities_cluster[batch_dates_n_quantities_cluster['batch_date'] > test_year] \
                           ['quantities'],
                       name='Optimized batches - actual values',
                       width=1e3 * pd.Timedelta(self.max_waiting_time).total_seconds() / 2))

            fig.add_trace(go.Bar(x=batch_dates_n_predicted_quantities_cluster[
                batch_dates_n_predicted_quantities_cluster['batch_date'] > test_year] \
                ['batch_date'],
                                 y=batch_dates_n_predicted_quantities_cluster[
                                     batch_dates_n_predicted_quantities_cluster['batch_date'] > test_year] \
                                     ['predicted_quantities'],
                                 name='Predicted batches',
                                 width=1e3 * pd.Timedelta(self.max_waiting_time).total_seconds() / 2,
                                 marker_color='lightgreen'))

            # Edit the layout
            fig.update_layout(
                title='Cluster #{} optimal batches vs predicted batches for the test period'.format(cluster),
                xaxis_title='Date',
                yaxis_title='Quantities')
            fig.show()

            sns.set(style="white")
            # Show the joint distribution using kernel density estimation
            try:
                g = sns.jointplot(
                    pd.Series(full_predictions['error_days'].values / (24 * 60 * 60 * 1e9), name='error_days\nin days'),
                    pd.Series(full_predictions['error_quantities'].values, name='error_quantities\nin%'),
                    kind="kde", height=7, space=0)
            except:
                print('Data is too small for this cluster to output the performance graph')

        return (batch_dates_n_quantities_cluster,
                batch_dates_n_predicted_quantities_cluster,
                full_predictions)

    def transform_the_optimized_df(self,
                                   sales_df,
                                   quantities):
        local_df = sales_df.reset_index()
        batch_dates_n_quantities_cluster = local_df.groupby('batch_date').sum()
        batch_dates_n_quantities_cluster['product_codes_quantities_and_dates'] = local_df. \
            groupby('batch_date')[['product_code', 'quantity', 'date']].apply(lambda x: x.values.tolist())
        batch_dates_n_quantities_cluster = batch_dates_n_quantities_cluster[['quantity',
                                                                             'product_codes_quantities_and_dates']]
        batch_dates_n_quantities_cluster.rename(index=str,
                                                columns={"quantity": quantities},
                                                inplace=True)

        return batch_dates_n_quantities_cluster
