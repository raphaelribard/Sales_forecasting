import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import plotly.graph_objects as go


def print_msg(msg):
    length=len(msg)
    print(length*'-'+'\n'+msg+'\n'+length*'-')
    return

def optimize_orders_processing(sales_clusters_df,cluster,maximum_waiting_time):
    
    sales_cluster=sales_clusters_df[sales_clusters_df['Cluster']==cluster][['noisy_date','noisy_quantity']].\
sort_values(by='noisy_date')
    
    sales_cluster=sales_cluster.reset_index(drop=True)
    
    max_wait_time=maximum_waiting_time.split(' ')[0]+ maximum_waiting_time.split(' ')[1][0]
    
    sales_cluster=sales_cluster.set_index('noisy_date')
    
    sales_cluster['number_of_previous_neighbours']=sales_cluster.rolling(max_wait_time).count()
    
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