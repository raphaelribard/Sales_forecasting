import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from fbprophet import Prophet
from fbprophet.plot import plot_plotly

import plotly.offline as py
import plotly.graph_objs as go


# Optimization part

def print_msg(msg):
    length=len(msg)
    print(length*'-'+'\n'+msg+'\n'+length*'-')
    return

def optimize_orders_processing(sales_clusters_df,cluster,maximum_waiting_time):
    
    try:
        sales_cluster=sales_clusters_df[sales_clusters_df['Cluster']==cluster][['noisy_date','noisy_quantity','product']].\
    sort_values(by='noisy_date')
    
    except:
         sales_cluster=sales_clusters_df[sales_clusters_df['Cluster']==cluster][['noisy_date','noisy_quantity']].\
    sort_values(by='noisy_date')
    
    sales_cluster=sales_cluster.reset_index(drop=True)
    
    max_wait_time=maximum_waiting_time.split(' ')[0]+ maximum_waiting_time.split(' ')[1][0]
    
    sales_cluster=sales_cluster.set_index('noisy_date')
    
    sales_cluster['number_of_previous_neighbours']=sales_cluster[['noisy_quantity']].rolling(max_wait_time).count()
    
    sales_cluster['batch']=0
    sales_cluster['batch_date']=sales_cluster.index.values

    i=0
    batch_group=1

    for i in tqdm(range(sales_cluster.shape[0])) :

        win_edge=sales_cluster.sort_values('number_of_previous_neighbours',ascending=False).\
                                           index[i]

        if (sales_cluster.batch[(sales_cluster.index<=win_edge)
                   & (sales_cluster.index>(win_edge-pd.Timedelta(maximum_waiting_time)))].sum())==0:

    #         print(i)
            sales_cluster.batch.where(~((sales_cluster.index<=win_edge)
                   & (sales_cluster.index>(win_edge-pd.Timedelta(maximum_waiting_time)))),
                                batch_group,
                                inplace=True)


            win=sales_cluster.batch_date[((sales_cluster.index<=win_edge)
                   & (sales_cluster.index>(win_edge-pd.Timedelta(maximum_waiting_time))))]
            
            batch_date=win_edge - ((win.iloc[-1]-win.iloc[0])/2)

            sales_cluster.batch_date[((sales_cluster.index<=win_edge)

                   & (sales_cluster.index>(win_edge-pd.Timedelta(maximum_waiting_time))))]=batch_date


            batch_group=batch_group+1

        i=i+1
        
    reduce_workload=100*(1-(sales_cluster.batch.value_counts().shape[0]/sales_cluster.shape[0]))
    
    print_msg('The workload is reduced by {}%'.format(str(int(reduce_workload))))
    
    batch_dates_df=sales_cluster[['batch','batch_date']].reset_index(drop=True).drop_duplicates()
    
    batch_dates_df['batch']=np.max(sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_quantity'].values)
    
    fig = go.Figure()

    
    fig.add_trace(go.Scatter(x=sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_date'].astype(str).values,
                             y=sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_quantity'].values,
                        mode='markers',
                        name='Cluster '+str(cluster)))

    fig.add_trace(go.Bar(x=batch_dates_df['batch_date'], y=batch_dates_df['batch'],
                    marker_color='lightgrey',
                         name='Batch dates'))

    for batch in sales_cluster.batch.value_counts().index.values:

        fig.add_trace(go.Scatter(x=sales_cluster[sales_cluster['batch']==batch].index.astype(str).values,
                             y=sales_cluster[sales_cluster['batch']==batch]['noisy_quantity'].values,
                        mode='markers',
                        name='batch '+str(batch)))




    # Edit the layout
    fig.update_layout(title='Optimized batches for cluster#'+str(cluster),
                       xaxis_title='Date',
                       yaxis_title='Quantities')

    fig.update_yaxes(range=[np.min(sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_quantity'].values),
                            np.max(sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_quantity'].values)])


    fig.show()
    
    return sales_cluster


#forecast_part

def generate_models_n_preds_df_n_plotly_viz(sales_clusters_df,test_start_date,product):
    """ This function aims to take a product and output:
        -  1 date model
        -  1 qty model
        -  1 predictions df
        -  1 plotly graph of the predictions
    """
    test_year=test_start_date.split('-')[0]
    # First select the product df with the sales dates and copy it into a new df
    prophet_date_df=sales_clusters_df[sales_clusters_df.product_code==product][['noisy_date']].copy()
    prophet_date_df.reset_index(drop=True,inplace=True)
    
    # Using global variable "test_year", generate the index where we split the df into a train year and a test year
    idx_split=prophet_date_df['noisy_date'][(prophet_date_df['noisy_date']>=test_year).cumsum()==1].index[0]
    
    # Generate the next sale date column which will be necessary to generate the "number of days to next sale" feature
    # This is the feature the "date model" will have to predict
    prophet_date_df=generate_next_sale_column(prophet_date_df)
    
    # Refactor the column names for the Facebook Prophet library
    prophet_date_df.rename(index=str, columns={"noisy_date": "ds"},
                 inplace=True)
    
    # Create the "date model"
    next_sale_date_model=Prophet()
    
    # Train the model on the training year
    print_msg('--> Now training the "date model"')
    next_sale_date_model.fit(prophet_date_df[:idx_split-1])
    
    # Make predictions on the complete df (training + test year) : this will let us check the overfitting later on
    next_sale_date_forecast = next_sale_date_model.predict(pd.DataFrame(prophet_date_df['ds']))
    
    # This is if you want to plot the date plot
    #plot_date_preds(next_sale_date_forecast,product,idx_split)
    
    # Select the product df with the sales dates AND quantities and copy it into a new df
    prophet_qty_df=sales_clusters_df[sales_clusters_df.product_code==product][['noisy_date','noisy_quantity']].copy()
    
    # Make it prophet compliant
    prophet_qty_df.rename(index=str, 
                          columns={"noisy_date": "ds",
                                      "noisy_quantity": "y"},
                         inplace=True)
    
    # Instantiate the quantity model 
    qty_model=Prophet()
    
    print_msg('--> Now training the "quantity model"')
    qty_model.fit(prophet_qty_df[:idx_split])
    
    # Make predictions on the complete df (training + test year) : this will let us check the overfitting later on
    qty_forecast = qty_model.predict(pd.DataFrame(prophet_qty_df['ds']))
    
    # Select the interesting columns for the qty df
    qty_predictions=qty_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Renaming the prophet output columns to identify the quantity predictions (in comparison with the sales dates preds)
    for col in qty_predictions.columns[1:]:
        qty_predictions.rename(index=str,
                                columns={col: col+'_qty'},
                                inplace=True)
    
    # Select the interesting columns for the sales df  
    next_sale_dates_predictions=next_sale_date_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    
    # Make the sales df readable and easy to identify
    for col in next_sale_dates_predictions.columns[1:]:
        next_sale_dates_predictions[col]=[pd.Timedelta(str(int(t))+' days') for t in next_sale_dates_predictions[col].values]
        next_sale_dates_predictions[col]=next_sale_dates_predictions['ds']+next_sale_dates_predictions[col]
        next_sale_dates_predictions.rename(index=str,
                                            columns={col: col+'_date'},
                                            inplace=True)
    
    # Prepare the full_preds df which will include dates, quuantities predictions and the actual values 
    #(dates and quantities)    
    full_predictions_df=qty_predictions.copy()
    # Sales dates and quantities concatenation
    for col in next_sale_dates_predictions.columns[1:]:
        local_list=next_sale_dates_predictions[col].values.tolist()
        local_list.insert(0,np.nan)
        full_predictions_df[col]=pd.to_datetime(local_list)
    
    # Include the actual values in the full_preds_df
    full_predictions_df['y']=prophet_qty_df['y'].values
    
    # Plot it (fancy with plotly)
    plot_full_preds(full_predictions_df,
                  product,
                  idx_split)
    
    return (next_sale_date_model,
            qty_model,
            full_predictions_df)

def generate_next_sale_column(df):
    df['next_sale_date']=df['noisy_date'].shift(-1)
    df.drop([len(df)-1],inplace=True)
    
    # y will be the number of days until the next sale
    df['y']=(df['next_sale_date']-df['noisy_date']).dt.days.astype(int)
    
    return df

def plot_full_preds(full_preds_df, product, idx_split):
    
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=full_preds_df['ds'],
                             y=full_preds_df['y'],
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

    fig.add_trace(go.Scatter(x=full_preds_df['yhat_date'][idx_split:],
                             y=full_preds_df['yhat_qty'][idx_split:].astype(int),
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


    fig.add_trace(go.Scatter(x=full_preds_df['yhat_lower_date'][idx_split:],
                             y=full_preds_df['yhat_lower_qty'][idx_split:].astype(int),
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

    fig.add_trace(go.Scatter(x=full_preds_df['yhat_upper_date'][idx_split:],
                             y=full_preds_df['yhat_upper_qty'][idx_split:].astype(int),
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
                x0=full_preds_df['ds'][idx_split-1],
                y0=np.min(full_preds_df['y']),
                x1=full_preds_df['ds'][idx_split-1],
                y1=np.max(full_preds_df['y']),
                line=dict(
                    color="grey",
                    width=2,
                    dash="dashdot"
                )
            )
        ]
    )

    fig.add_trace(go.Scatter(
        x=[full_preds_df['ds'][idx_split-2],
          full_preds_df['ds'][idx_split]],
        y=[np.mean(full_preds_df['y']),
                   np.mean(full_preds_df['y'])],
        text=["Training data <--",
              "--> Testing data"],
        mode="text",
        name='training and test data'
    ))

    # Edit the layout
    fig.update_layout(title='Predicted dates and quantities for ' + product,
                       xaxis_title='Date',
                       yaxis_title='Sales quantities')



    fig.show()
    
    return

def get_aggregated_preds(sales_clusters_df,cluster,test_start_date):
    
    cluster_products=sales_clusters_df[sales_clusters_df.Cluster==cluster].product_code.unique()
    
    cluster_preds=[]
    for prod in cluster_products:
        cluster_preds.append(generate_models_n_preds_df_n_plotly_viz(sales_clusters_df,test_start_date,prod))
        
    cluster_aggregated_preds=cluster_preds[0][2]
    cluster_aggregated_preds['product']=cluster_products[0]
    for i,prod in enumerate(cluster_products[1:]):
        print(i)
        print(prod)
        local_df=cluster_preds[i+1][2]
        local_df['product']=prod
        cluster_aggregated_preds=pd.concat([cluster_aggregated_preds,local_df])
        
    cluster_preds_test_year=cluster_aggregated_preds[cluster_aggregated_preds['yhat_date']>test_start_date]
    

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_date'].astype(str).values,
                                 y=sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_quantity'].values,
                            mode='markers',
                            name='Cluster '+str(cluster)))

    fig.add_trace(go.Scatter(x=cluster_preds_test_year['yhat_date'].astype(str).values,
                                 y=cluster_preds_test_year['yhat_qty'].values,
                            mode='markers',
                            name='Cluster '+str(cluster)+ ' preds'))

    fig.update_layout(
            shapes=[
                # Line Vertical
                go.layout.Shape(
                    type="line",
                    x0=test_start_date,
                    y0=np.min(sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_quantity'].values),
                    x1=test_start_date,
                    y1=np.max(sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_quantity'].values),
                    line=dict(
                        color="grey",
                        width=2,
                        dash="dashdot"
                    )
                )
            ]
        )

    fig.add_trace(go.Scatter(
            x=[str(int(test_start_date.split('-')[0])-1)+'-10-10',
                  test_start_date.split('-')[0]+'-03-10'],
            y=[np.min(sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_quantity'].values),
                       np.min(sales_clusters_df[sales_clusters_df['Cluster']==cluster]['noisy_quantity'].values)],
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
    
    return cluster_aggregated_preds