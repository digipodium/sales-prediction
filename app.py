""" Main Module of sales prediction app"""
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from PIL import Image

#do not show warnings
import warnings
warnings.filterwarnings("ignore")

# Import statsmodels.formula.api
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pickle

#import Keras
from tensorflow import keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM

#import machine learning algorithm
import xgboost as xgb
from xgboost import plot_importance
from sklearn.preprocessing import LabelEncoder,  MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, train_test_split



# Function to Load Data for sales prediction-Online Retail Data
@st.cache(allow_output_mutation=True)
def load_data():
    retail_data = pd.read_csv('Data/OnlineRetail.csv', encoding='unicode_escape')
    return retail_data
    
data_load = load_data() # calling data loading function
data = data_load

#Main heading
main_heading = """
                <div>
                <h1 style="background-color: CadetBlue;text-align:center;">Sales Prediction Of Products</h1>
                <h2 style="text-align:right;"><i>Using Regression Algorithms</i></h2>
                </div>
            """
st.markdown(main_heading, unsafe_allow_html=True)
st.write("---")

#Main Function
def main():

    """Main Function of Sales Prediction"""
    option = st.sidebar.radio("Use Navigation Here:",("Home", "Data Metrics",
                        "Customer Segmentation","Customer Value Prediction",
                        "Churn Prediction", "Next Purchase Day", "Predicting Sales",
                        "Market Response"
                         ))
    if option == "Home":
        with st.spinner("Wait for it..."):
            st.write(home_page())
    elif option == "Data Metrics":
        with st.spinner("Wait for it..."):
            st.write(metrics_of_data())
    elif option == "Customer Segmentation":
        with st.spinner("In Progress"):
            st.write(customer_segmentation())
    elif option == "Customer Value Prediction":
        with st.spinner("In Progress"):
            st.write(customer_value_prediction())
    elif option == "Churn Prediction":
        with st.spinner("In Progress"):
            st.write(churn_prediction())
    elif option == "Next Purchase Day":
        with st.spinner("In Progress"):
            st.write(predict_next_purchase_day())
    elif option == "Predicting Sales":
        with st.spinner("In Progress"):
            st.write(predicting_sales())
    elif option == "Market Response":
        with st.spinner("In Progress"):
            st.write(market_response_model())


def home_page():
    st.title("Overview Of Project:")
    st.write("**_“No great marketing decisions have ever been made on qualitative data.” – John Sculley_**")
    st.write("""
    _In this project, we are addressing the problem of sales prediction or forecasting of an item on customer’s future demand in different stores
    across various locations and products based on the previous record. Different machine learning algorithms like linear regression analysis, random forest, etc
    are used for prediction or forecasting of sales volume._
    """)

    st.write("""
    _This project, helps in developing the strategies of business about the marketplace to improve the knowledge of market. A standard sales prediction study can
    help in deeply analyzing the situations or the conditions previously occurred and then, the inference can be applied about customer acquisition, funds inadequacy
    and strengths before setting a budget and marketing plans for the upcoming year._
    """)
    

#function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final       


# function for Data Metrics
def metrics_of_data():
    st.header("**_Metrics of Data_**")
    st.write('The metrics of data will help to define the core information of data')
    st.latex(r'''Revenue = Active Customer count * Order Count * Average Revenue Per Order''')
    #converting the type of Invoice Date Field from string to datetime.
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

    #creating YearMonth field for the ease of reporting and visualization
    data['InvoiceYearMonth'] = data['InvoiceDate'].map(lambda date: 100*date.year + date.month)

    #calculate Revenue for each row and create a new dataframe with YearMonth - Revenue columns
    data['Revenue'] = data['UnitPrice'] * data['Quantity']
    revenue = data.groupby(['InvoiceYearMonth'])['Revenue'].sum().reset_index()

    if st.checkbox('View Data'):
        st.dataframe(data.head(10))
    
    #X and Y axis inputs for Plotly graph. We use Scatter for line graphs
    plot_data = [
    go.Scatter(
            x=revenue['InvoiceYearMonth'],
            y=revenue['Revenue'],
    )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category", 'title': "YearMonth"},
            yaxis= {'title': "Revenue"},
            title='Montly Revenue'
    )
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)      #Monthly Rvenue Visualization
    st.write('This graph clearly shows our revenue is growing especially Aug ‘11 onwards')

    st.write("Now, Let's figure out what is our **_Monthly Revenue Growth Rate:_**")
    #using pct_change() function to see monthly percentage change
    revenue['MonthlyGrowth'] = revenue['Revenue'].pct_change()

    #code to show Data

    #visualization - line graph
    plot_data = [
        go.Scatter(
            x=revenue.query("InvoiceYearMonth < 201112")['InvoiceYearMonth'],
            y=revenue.query("InvoiceYearMonth < 201112")['MonthlyGrowth'],
        )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category", 'title': "YearMonth"},
            yaxis= {'title': "Growth in Revenue"},
            title='Montly Growth Rate'
        )

    mon_revenue_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(mon_revenue_fig)

    st.write("""Looks good, but we need to figure out what happen on April.
    For this we have to do deep analysis like **Monthly Active Customers**, 
    by counting unique customersID""")
    
    #creating a new dataframe with UK customers only
    uk = data.query("Country=='United Kingdom'").reset_index(drop=True)

    #creating monthly active customers dataframe by counting unique Customer IDs
    monthly_active = uk.groupby('InvoiceYearMonth')['CustomerID'].nunique().reset_index()

    #print the dataframe
    if st.checkbox('Monthly Active Customers Data'):
        st.dataframe(monthly_active.head(10))
    

    #plotting the output
    plot_data = [
        go.Bar(
            x=monthly_active['InvoiceYearMonth'],
            y=monthly_active['CustomerID'],
        )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category", 'title': "YearMonth"},
            yaxis= {'title': "Customers"},
            title='Monthly Active Customers'
        )

    mon_active_cust_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(mon_active_cust_fig)

    st.write("In _April_, Monthly Active Customer number dropped to 817 from 923 (-11.5%).")
    #create a new dataframe for no. of order by using quantity field
    monthly_sales = uk.groupby('InvoiceYearMonth')['Quantity'].sum().reset_index()

    #print the dataframe
    if st.checkbox('Monthly Sales Data'):
        monthly_sales.head(10)

    #plot
    plot_data = [
        go.Bar(
            x=monthly_sales['InvoiceYearMonth'],
            y=monthly_sales['Quantity'],
        )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category", 'title': "YearMonth"},
            yaxis= {'title': "Order Quantity"},
            title='Monthly Total Order'
        )

    order_count_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(order_count_fig)
    st.write("""Order Count is also declined in April (279k to 257k, -8%).
     Active Customer Count directly affected Order Count decrease. 
    For this we should definitely check our Average Revenue per Order as well.""")
    # create a new dataframe for average revenue by taking the mean of it
    monthly_order_avg = uk.groupby('InvoiceYearMonth')['Revenue'].mean().reset_index()

    #print the dataframe
    if st.checkbox('Monthly Order Average Data'):
        monthly_order_avg.head(10)

    #plot the bar chart
    plot_data = [
        go.Bar(
            x=monthly_order_avg['InvoiceYearMonth'],
            y=monthly_order_avg['Revenue'],
        )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category", 'title': "YearMonth"},
            yaxis= {'title': "Avg. Monthly Order"},
            title='Monthly Order Average'
        )
    monthly_oreder_avg_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(monthly_oreder_avg_fig)

    st.write("Even the monthly order average dropped for April (16.7 to 15.8). We observed slow-down in every metric.")

    html_1 = """
            <p>
            Let’s continue investigating some other important metrics:</p>
            <ul>
                <li><b>New Customer Ratio:</b> a good indicator of if
                    we are losing our existing customers or unable to attract new ones</li>
                <li><b>Retention Rate:</b> Indicates how many customers we retain over specific time window.
                    We will be showing examples for monthly retention rate and cohort based retention rate.</li>
            </ul>
            """
    st.write(html_1,unsafe_allow_html = True)

    st.write("""**New Customer Ratio:** First we should define what is a new customer.
     In our dataset, we can assume a new customer is whoever did his/her first purchase in the time window we defined.
    We will be using .min() function to find our first purchase date for each customer and define new customers based on that.""")
    #create a dataframe contaning CustomerID and first purchase date
    min_purchase = uk.groupby('CustomerID').InvoiceDate.min().reset_index()
    min_purchase.columns = ['CustomerID','MinPurchaseDate']
    min_purchase['MinPurchaseYearMonth'] = min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)

    #merge first purchase date column to our main dataframe (tx_uk)
    uk = pd.merge(uk, min_purchase, on='CustomerID')


    #create a column called User Type and assign Existing 
    #if User's First Purchase Year Month before the selected Invoice Year Month
    uk['UserType'] = 'New'
    uk.loc[uk['InvoiceYearMonth']>uk['MinPurchaseYearMonth'],'UserType'] = 'Existing'

    #calculate the Revenue per month for each user type
    user_type_revenue = uk.groupby(['InvoiceYearMonth','UserType'])['Revenue'].sum().reset_index()

    #filtering the dates and plot the result
    user_type_revenue = user_type_revenue.query("InvoiceYearMonth != 201012 and InvoiceYearMonth != 201112")
    plot_data = [
        go.Scatter(
            x=user_type_revenue.query("UserType == 'Existing'")['InvoiceYearMonth'],
            y=user_type_revenue.query("UserType == 'Existing'")['Revenue'],
            name = 'Existing'
        ),
        go.Scatter(
            x=user_type_revenue.query("UserType == 'New'")['InvoiceYearMonth'],
            y=user_type_revenue.query("UserType == 'New'")['Revenue'],
            name = 'New'
        )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category", 'title': "YearMonth"},
            yaxis= {'title': "Revenue"},
            title='New vs Existing'
        )
    new_existing_cust_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(new_existing_cust_fig)
    st.write("""Existing customers are showing a positive trend and tell us that our customer base is growing but new customers have a slight negative trend.""")

    st.write("""Let’s have a better view by looking at the New Customer Ratio:""")
    #create a dataframe that shows new user ratio - we also need to drop NA values (first month new user ratio is 0)
    user_ratio = uk.query("UserType == 'New'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique()/uk.query("UserType == 'Existing'").groupby(['InvoiceYearMonth'])['CustomerID'].nunique() 
    user_ratio = user_ratio.reset_index()
    user_ratio = user_ratio.dropna()

    #print the dafaframe
    if st.checkbox('User Ratio Data'):
       user_ratio.head(10)
    

    #plot the result

    plot_data = [
        go.Bar(
            x=user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['InvoiceYearMonth'],
            y=user_ratio.query("InvoiceYearMonth>201101 and InvoiceYearMonth<201112")['CustomerID'],
        )
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category", 'title': "YearMonth"},
            yaxis= {'title': "Customers Ratio"},
            title='New Customer Ratio'
        )
    new_customer_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(new_customer_fig)
    st.write("""New Customer Ratio has declined as expected (we assumed on Feb, all customers were New) and running around 20%.""")

    st.write("""**Monthly Retention Rate:** Retention rate should be monitored very closely because it indicates how sticky is your service
        and how well your product fits the market.For making Monthly Retention Rate visualized, we need to calculate how many customers
        retained from previous month.""")
    st.latex(r"""Monthly Retention Rate = Retained Customers From Prev. Month/Active Customers Total""")
    st.write("""We will be using crosstab() function of pandas which makes calculating Retention Rate super easy.""")
    #identify which users are active by looking at their revenue per month
    user_purchase = uk.groupby(['CustomerID','InvoiceYearMonth'])['Revenue'].sum().reset_index()

    #create retention matrix with crosstab
    retention = pd.crosstab(user_purchase['CustomerID'], user_purchase['InvoiceYearMonth']).reset_index()

    if st.checkbox('Retention Data'):
        retention.head(10)

    #create an array of dictionary which keeps Retained & Total User count for each month
    months = retention.columns[2:]
    retention_array = []
    for i in range(len(months)-1):
        retention_data = {}
        selected_month = months[i+1]
        prev_month = months[i]
        retention_data['InvoiceYearMonth'] = int(selected_month)
        retention_data['TotalUserCount'] = retention[selected_month].sum()
        retention_data['RetainedUserCount'] = retention[(retention[selected_month]>0) & (retention[prev_month]>0)][selected_month].sum()
        retention_array.append(retention_data)
        
    #convert the array to dataframe and calculate Retention Rate
    retention = pd.DataFrame(retention_array)
    retention['RetentionRate'] = retention['RetainedUserCount']/retention['TotalUserCount']

    #plot the retention rate graph
    plot_data = [
        go.Scatter(
            x=retention.query("InvoiceYearMonth<201112")['InvoiceYearMonth'],
            y=retention.query("InvoiceYearMonth<201112")['RetentionRate'],
            name="organic"
        )
        
    ]

    plot_layout = go.Layout(
            xaxis={"type": "category", 'title': "YearMonth"},
            yaxis= {'title': "Retention Rate per Month"},
            title='Monthly Retention Rate'
        )
    monthly_retention_rate_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(monthly_retention_rate_fig)
    st.write("""Monthly Retention Rate significantly jumped from June to August and went back to previous levels afterwards.""")

    if st.checkbox("See DataFrame"):
        data.head()


# Function for customer Segmentation
def customer_segmentation():
    st.header("**_Customer Segmentation_**")
    st.write("""We have analyzed the major **_metrics_** for our online retail business.  
    It’s time to focus on customers and segment them.""")

    st.write("""
    <p>RFM stands for <b><i>Recency - Frequency - Monetary Value</i></p>. 
    Theoretically we will have segments like below:</p>
    <ul>
    <li><b>Low Value:</b> Customers who are less active than others, not very frequent buyer/visitor and generates 
    very low - zero - maybe negative revenue.</li>
    <li><b>Mid Value:</b>Often using our platform (but not as much as our High Values),
    fairly frequent and generates moderate revenue.</li>
    <li><b>High Value:</b> The group we don’t want to lose. High Revenue, Frequency and low Inactivity.</li></ul>
    <p>As the methodology, we need to calculate Recency, Frequency and Monetary Value
    and apply unsupervised machine learning to identify different groups (clusters) for each. 
    Let’s jump to find RFM Clustering.</p>
    """, unsafe_allow_html = True)

    st.write("""**_Recency:_** To calculate recency, we need to find out most recent purchase date of each customer and 
    see how many days they are inactive for. After having no. of inactive days for each customer, 
    we will apply **K-means** clustering to assign customers a recency score.""")

    #convert the string date field to datetime
    data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])

    #we will be using only UK data
    uk = data.query("Country=='United Kingdom'").reset_index(drop=True)
    #create a generic user dataframe to keep CustomerID and new segmentation scores
    user = pd.DataFrame(data['CustomerID'].unique())
    user.columns = ['CustomerID']

    #get the max purchase date for each customer and create a dataframe with it
    max_purchase = uk.groupby('CustomerID').InvoiceDate.max().reset_index()
    max_purchase.columns = ['CustomerID','MaxPurchaseDate']

    #we take our observation point as the max invoice date in our dataset
    max_purchase['Recency'] = (max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days

    #merge this dataframe to our new user dataframe
    user = pd.merge(user, max_purchase[['CustomerID','Recency']], on='CustomerID')
    if st.checkbox('View Dataframe'):
        st.write(user.head(5))

    if st.checkbox('Recency Data Describe'):
        st.write(user.Recency.describe())

    #plot a recency histogram

    plot_data = [
        go.Histogram(
            x=user['Recency']
        )
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Users"},
            xaxis= {'title': "Recency"},
            title='Recency'
        )
    recency_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(recency_fig)

    st.write("""By using K-means clustering to assign a recency score. 
    To find it out, no of clusters for K-mean algorithm we will apply Elbow Method.
    **Elbow Method** simply tells the optimal cluster number for optimal inertia.
    """)
    #figure to find cluster
    sse={}
    recency = user[['Recency']]
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(recency)
        recency["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_ 
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    st.pyplot(plt.show())
    

    st.write("""In above it looks like 3 is the optimal one. Based on business requirements, 
    we can go ahead with less or more clusters. We will be selecting 4 for this example:""")

    #build 4 clusters for recency and add it to dataframe
    #no_clster = st.slider("Select size of cluster:", )
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(user[['Recency']])
    user['RecencyCluster'] = kmeans.predict(user[['Recency']])

    user = order_cluster('RecencyCluster', 'Recency',user,False)
    #tx_user.groupby('FrequencyCluster')['Frequency'].describe()

    st.write(user.groupby('RecencyCluster')['Recency'].describe())

    # for Frequency
    st.write(""" 
        **Frequency:**  To create frequency clusters, 
        we need to find total number orders for each customer
    """)

    #get order counts for each user and create a dataframe with it
    frequency = uk.groupby('CustomerID').InvoiceDate.count().reset_index()
    frequency.columns = ['CustomerID','Frequency']

    #add this data to our main dataframe
    user = pd.merge(user, frequency, on='CustomerID')

    #plot the histogram
    plot_data = [
        go.Histogram(
            x=user.query('Frequency < 1000')['Frequency']
        )
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "No. of Orders"},
            xaxis= {'title': "Frequency"},
            title='Frequency'
        )
    frequebcy_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(frequebcy_fig)

    #same logic for having frequency clusters and assign this to each customer
    st.write("""Apply the same logic for having frequency clusters and assign this to each customer:""")
    #k-means
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(user[['Frequency']])
    user['FrequencyCluster'] = kmeans.predict(user[['Frequency']])

    #order the frequency cluster
    user = order_cluster('FrequencyCluster', 'Frequency',user,True)

    #see details of each cluster
    st.write(user.groupby('FrequencyCluster')['Frequency'].describe())

    st.write("""**_As the same notation as recency clusters, 
    high frequency number indicates better customers._**""")

    # for Revenue
    st.write("""**Revenue: ** Calculate revenue for each customer and plot a histogram and apply the same clustering method.""")
    
    
    #calculate revenue for each customer
    uk['Revenue'] = uk['UnitPrice'] * uk['Quantity']
    revenue = uk.groupby('CustomerID').Revenue.sum().reset_index()

    #merge it with our main dataframe
    user = pd.merge(user, revenue, on='CustomerID')

    #plot the histogram
    plot_data = [
        go.Histogram(
            x=user.query('Revenue < 10000')['Revenue']
        )
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "User"},
            xaxis= {'title': "Revenue"},
            title='Monetary Value'
        )
    revenue_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(revenue_fig)

    st.write("""We have some customers with negative revenue as well. 
    Let’s continue and apply k-means clustering:""")
    #apply clustering
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(user[['Revenue']])
    user['RevenueCluster'] = kmeans.predict(user[['Revenue']])


    #order the cluster numbers
    user = order_cluster('RevenueCluster', 'Revenue',user,True)

    #show details of the dataframe
    user.groupby('RevenueCluster')['Revenue'].describe()
    st.write("**_Overall Score_**", "Awesome! We have scores (cluster numbers) for recency, frequency & revenue. Let’s create an overall score out of them:")

    #calculate overall score and use mean() to see details
    user['OverallScore'] = user['RecencyCluster'] + user['FrequencyCluster'] + user['RevenueCluster']
    user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()
    st.write("""_The scoring above clearly shows us that customers with score 8 is our best customers whereas 0 is the worst._""")

    #cluster Info
    cluster_info = """

        <p>using Score values</p>
        <ul>
            <li>0-2 : Low Value</li>
            <li>3-4 : Mid Value</li>
            <li>5+  : High Value</li>
        </ul> 
        """
    st.markdown(cluster_info,unsafe_allow_html=True)
    #Dataframe
    user['Segment'] = 'Low-Value'
    user.loc[user['OverallScore']>2,'Segment'] = 'Mid-Value' 
    user.loc[user['OverallScore']>4,'Segment'] = 'High-Value' 

    st.write('Showing segments with **Scatter Plot** Daigram')
    #snippet For scatterplot
    #Revenue vs Frequency
    graph = user.query("Revenue < 50000 and Frequency < 2000")

    plot_data = [
        go.Scatter(
            x=graph.query("Segment == 'Low-Value'")['Frequency'],
            y=graph.query("Segment == 'Low-Value'")['Revenue'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=graph.query("Segment == 'Mid-Value'")['Frequency'],
            y=graph.query("Segment == 'Mid-Value'")['Revenue'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=graph.query("Segment == 'High-Value'")['Frequency'],
            y=graph.query("Segment == 'High-Value'")['Revenue'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Revenue"},
            xaxis= {'title': "Frequency"},
            title='Segments-Revenue vs Frequency'
        )
    fig_1 = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig_1)

    #Revenue Recency

    graph = user.query("Revenue < 50000 and Frequency < 2000")

    plot_data = [
        go.Scatter(
            x=graph.query("Segment == 'Low-Value'")['Recency'],
            y=graph.query("Segment == 'Low-Value'")['Revenue'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=graph.query("Segment == 'Mid-Value'")['Recency'],
            y=graph.query("Segment == 'Mid-Value'")['Revenue'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=graph.query("Segment == 'High-Value'")['Recency'],
            y=graph.query("Segment == 'High-Value'")['Revenue'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Revenue"},
            xaxis= {'title': "Recency"},
            title='Segments: Revenue vs Recency'
        )
    fig_2 = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig_2)

    # Revenue vs Frequency
    graph = user.query("Revenue < 50000 and Frequency < 2000")

    plot_data = [
        go.Scatter(
            x=graph.query("Segment == 'Low-Value'")['Recency'],
            y=graph.query("Segment == 'Low-Value'")['Frequency'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=graph.query("Segment == 'Mid-Value'")['Recency'],
            y=graph.query("Segment == 'Mid-Value'")['Frequency'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=graph.query("Segment == 'High-Value'")['Recency'],
            y=graph.query("Segment == 'High-Value'")['Frequency'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Frequency"},
            xaxis= {'title': "Recency"},
            title='Segments: Frequency vs Recency'
        )
    fig_3 = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig_3)


    #segment Info
    rfm_info = """

        <p><i>You can see how the segments are clearly differentiated from each other in terms of RFM in graph above.</i></p>

        <p>Strategies can me made clearly using these segmentation.Actions can be taken</p>

        <ul>
            <li>Low Value : Increase Frequency</li>
            <li>Mid Value : Improve Retention + Increase Frequency</li>
            <li>High Value : Improve Retention</li>
        </ul> 
        """
    st.markdown(rfm_info,unsafe_allow_html=True)
    st.write(user.head(10))


#Customers Lifetime Value Prediction
def customer_value_prediction():
    st.header("**_Customer Lifetime Value_**")
    st.write("""
        We invest in customers (acquisition costs, offline ads, promotions, discounts & etc.) 
        to generate revenue and be profitable. Naturally, these actions make some customers super 
        valuable in terms of lifetime value but there are always some customers who pull down the profitability. 
        We need to identify these behavior patterns, segment customers and act accordingly.
       By using the **equation below**, we can have Lifetime Value for each customer in that specific time window:
    """)
    if st.checkbox('Show DAtaframe'):
        st.write(data.head(5))
    st.latex(r'''Lifetime Value: Total Gross Revenue - Total Cost''')
    # data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
    uk = data.query("Country=='United Kingdom'").reset_index(drop=True)
    

    #create 3m and 6m dataframes
    # df_3m = uk[(uk.InvoiceDate < date(2011,6,1)) & (uk.InvoiceDate >= date(2011,3,1))].reset_index(drop=True)
    # df_6m = uk[(uk.InvoiceDate >= date(2011,6,1)) & (uk.InvoiceDate < date(2011,12,1))].reset_index(drop=True)
    df_3m = uk[(uk.InvoiceYearMonth < 201106) & (uk.InvoiceYearMonth >= 201103)].reset_index(drop=True)
    df_6m = uk[(uk.InvoiceYearMonth >= 201106) & (uk.InvoiceYearMonth < 201212)].reset_index(drop=True)



    #create tx_user for assigning clustering
    user = pd.DataFrame(df_3m['CustomerID'].unique())
    user.columns = ['CustomerID']

    #calculate recency score
    max_purchase = df_3m.groupby('CustomerID').InvoiceDate.max().reset_index()
    max_purchase.columns = ['CustomerID','MaxPurchaseDate']
    max_purchase['Recency'] = (max_purchase['MaxPurchaseDate'].max() - max_purchase['MaxPurchaseDate']).dt.days
    user = pd.merge(user, max_purchase[['CustomerID','Recency']], on='CustomerID')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(user[['Recency']])
    user['RecencyCluster'] = kmeans.predict(user[['Recency']])

    user = order_cluster('RecencyCluster', 'Recency',user,False)

    #calcuate frequency score
    frequency = df_3m.groupby('CustomerID').InvoiceDate.count().reset_index()
    frequency.columns = ['CustomerID','Frequency']
    user = pd.merge(user, frequency, on='CustomerID')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(user[['Frequency']])
    user['FrequencyCluster'] = kmeans.predict(user[['Frequency']])

    user = order_cluster('FrequencyCluster', 'Frequency',user,True)

    #calcuate revenue score
    df_3m['Revenue'] = df_3m['UnitPrice'] * df_3m['Quantity']
    revenue = df_3m.groupby('CustomerID').Revenue.sum().reset_index()
    user = pd.merge(user, revenue, on='CustomerID')

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(user[['Revenue']])
    user['RevenueCluster'] = kmeans.predict(user[['Revenue']])
    user = order_cluster('RevenueCluster', 'Revenue',user,True)


    #overall scoring
    user['OverallScore'] = user['RecencyCluster'] + user['FrequencyCluster'] + user['RevenueCluster']
    user['Segment'] = 'Low-Value'
    user.loc[user['OverallScore']>2,'Segment'] = 'Mid-Value' 
    user.loc[user['OverallScore']>4,'Segment'] = 'High-Value' 


    #show data frame
    if st.checkbox('User Data'):
        user.head()

    #calculate revenue and create a new dataframe for it
    df_6m['Revenue'] = df_6m['UnitPrice'] * df_6m['Quantity']
    user_6m = df_6m.groupby('CustomerID')['Revenue'].sum().reset_index()
    user_6m.columns = ['CustomerID','m6_Revenue']


    #plot LTV histogram
    plot_data = [
        go.Histogram(
            x=user_6m.query('m6_Revenue < 10000')['m6_Revenue']
        )
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Revenue"},
            xaxis= {'title': "6-month Revenue"},
            title='6-month Revenue'
        )
    fig_6m = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig_6m)

    st.write("""Histogram clearly shows we have customers with negative LTV. 
    Filtering out the outliers makes sense to have a proper machine learning model.""")

    st.write("""Next merging 3 months and 6 months dataframes to see correlations
     between LTV and the feature set.""")

    df_merge = pd.merge(user, user_6m, on='CustomerID', how='left')
    df_merge = df_merge.fillna(0)

    df_graph = df_merge.query("m6_Revenue < 30000")

    plot_data = [
        go.Scatter(
            x=df_graph.query("Segment == 'Low-Value'")['OverallScore'],
            y=df_graph.query("Segment == 'Low-Value'")['m6_Revenue'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=df_graph.query("Segment == 'Mid-Value'")['OverallScore'],
            y=df_graph.query("Segment == 'Mid-Value'")['m6_Revenue'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=df_graph.query("Segment == 'High-Value'")['OverallScore'],
            y=df_graph.query("Segment == 'High-Value'")['m6_Revenue'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "6-month LTV"},
            xaxis= {'title': "RFM Score"},
            title='LTV'
        )
    fig_merge = go.Figure(data=plot_data, layout=plot_layout) 
    st.plotly_chart(fig_merge)

    st.write("""_Positive correlation is quite visible here. High RFM score means high LTV._""")

    st.write("""LTV itself is a regression problem. But here, we want LTV segments. 
    Because it makes it more actionable and easy to communicate with other people. 
    By applying K-means clustering, we can identify our existing LTV groups and build
    segments on top of it. for this we apply clustering and have 3 - segments:""")

    st.markdown("""
    <ol>
     <li> Low LTV</li>
     <li> High LTV</li>
     <li> Mid LTV</li>
    </ol>
    """, unsafe_allow_html= True)

    #remove outliers
    df_merge = df_merge[df_merge['m6_Revenue']<df_merge['m6_Revenue'].quantile(0.99)]


    #creating 3 clusters
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(df_merge[['m6_Revenue']])
    df_merge['LTVCluster'] = kmeans.predict(df_merge[['m6_Revenue']])

    #order cluster number based on LTV
    df_merge = order_cluster('LTVCluster', 'm6_Revenue',df_merge,True)

    #creatinga new cluster dataframe
    df_cluster = df_merge.copy()

    #see details of the clusters
    st.write(df_cluster.groupby('LTVCluster')['m6_Revenue'].describe())

    st.write("2 is the best with average 8.2k LTV whereas 0 is worst with 396.")

    st.markdown("""


    """,unsafe_allow_html=True)

    #convert categorical columns to numerical
    df_class = pd.get_dummies(df_cluster)

    #calculate and show correlations
    corr_matrix = df_class.corr()
    corr_matrix['LTVCluster'].sort_values(ascending=False)
    st.write(corr_matrix['LTVCluster'].sort_values(ascending=False))
    st.write("3 months _Revenue, Frequency and RFM_ scores will be helpful for our machine learning models.")

    #create X and y, X will be feature set and y is the label - LTV
    X = df_class.drop(['LTVCluster','m6_Revenue'],axis=1)
    y = df_class['LTVCluster']

    #split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)

    #XGBoost Multiclassification Model
    ltv_xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1).fit(X_train, y_train)

    st.write('Accuracy of XGB classifier on training set: {:.2f}'
        .format(100 * ltv_xgb_model.score(X_train, y_train)))
    st.write('Accuracy of XGB classifier on test set: {:.2f}'
        .format(100 * ltv_xgb_model.score(X_test[X_train.columns], y_test)))
    st.write("Accuracy shows 84% on the test set. Looks really good.")
    

    y_pred = ltv_xgb_model.predict(X_test)
    classification_report(y_test, y_pred)
    st.image(Image.open('classification_report.png'), caption="Inertia Graph", use_column_width=True)


    st.write("""Precision and recall are acceptable for 0. 
    As for cluster 0 (Low LTV), if model tells us this customer belongs
     to cluster 0, 90 out of 100 will be correct (precision).
     And the model successfully identifies 93% of actual cluster 0 customers (recall)""")

    st.write("""Great! Now we have a machine learning model which predicts 
    the future LTV segments of our customers. 
    We can easily adapt our actions based on that as 
    we definitely do not want to lose customers with high LTV""")


#churn Prediction
def churn_prediction():
    st.header('**_Churn Prediction_ **')
    st.write("""Since we know our best customers by segmentation and lifetime value prediction, we should also work hard on retaining them. 
    That’s what makes Retention Rate is one of the most critical metrics. Retention Rate is an indication of how good your product market fits. 
    If your product is not satisfactory, you should see your customers churning very soon. 
    One of the powerful tools to improve Retention Rate is Churn Prediction.
    By using this technique, you can easily find out who is likely to churn in the given period.""")
    churn_df = pd.read_csv("Data/churn_data.csv")
    st.write(churn_df.head())
    churn_df.loc[churn_df.Churn=='No','Churn'] = 0 
    churn_df.loc[churn_df.Churn=='Yes','Churn'] = 1
    churn_df.Churn = churn_df.Churn.astype('int')

    def plot_churn_graphs(datafieldname):

        df_plot = churn_df.groupby(datafieldname).Churn.mean().reset_index()
        plot_data = [
            go.Bar(
                x=df_plot[datafieldname],
                y=df_plot['Churn'],
                width = [0.5, 0.5],
                marker=dict(
                color=['green', 'blue', 'yellow', 'red'])
            )
        ]
        plot_layout = go.Layout(
                xaxis={"type": "category"},
                yaxis={"title": "Churn Rate"},
                title=datafieldname.upper(),
                plot_bgcolor  = 'rgb(243,243,243)',
                paper_bgcolor  = 'rgb(243,243,243)',
            )
        st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))
    
    plot_churn_graphs('gender')
    plot_churn_graphs('PhoneService')
    plot_churn_graphs('PaperlessBilling')
    plot_churn_graphs('PaymentMethod')
    
    def churn_scatter_graph(datafield1):
        df_plot = churn_df.groupby(datafield1).Churn.mean().reset_index()

        plot_data = [
            go.Scatter(
                x=df_plot[datafield1],
                y=df_plot['Churn'],
                mode='markers',
                name='Low',
                marker= dict(size= 7,
                    line= dict(width=1),
                    color= 'blue',
                    opacity= 0.8
                ),
            )
        ]

        plot_layout = go.Layout(
                yaxis= {'title': "Churn Rate"},
                xaxis= {'title': datafield1},
                title='{} based Churn rate'.format(datafield1),
                plot_bgcolor  = "rgb(243,243,243)",
                paper_bgcolor  = "rgb(243,243,243)",
            )
        return st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))

    churn_scatter_graph('tenure')

    
    def kmeans_churn_graph(datafieldname1, datafieldnam2, churn_df=churn_df):
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(churn_df[[datafieldname1]])
        churn_df[datafieldnam2] = kmeans.predict(churn_df[[datafieldname1]])
        churn_df = order_cluster(datafieldnam2, datafieldname1,churn_df,True)
        churn_df.groupby(datafieldnam2).tenure.describe()
        churn_df[datafieldnam2] = churn_df[datafieldnam2].replace(
                                            {0:'Low',1:'Mid',2:'High'})
        
        df_plot = churn_df.groupby(datafieldnam2).Churn.mean().reset_index()
        plot_data = [
            go.Bar(
                x=df_plot[datafieldnam2],
                y=df_plot['Churn'],
                width = [0.5, 0.5, 0.5,0.5],
                marker=dict(
                color=['green', 'blue', 'orange','red'])
            )
        ]

        plot_layout = go.Layout(
                xaxis={"type": "category","categoryarray":['Low','Mid','High'], 'title': datafieldnam2},
                yaxis={'title':'Churn Rate'},
                title='{} vs Churn Rate'.format(datafieldnam2),
                plot_bgcolor  = "rgb(243,243,243)",
                paper_bgcolor  = "rgb(243,243,243)",
            )
        return st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))
    
    kmeans_churn_graph('tenure', 'TenureCluster')

    df_plot = churn_df.copy()
    df_plot['MonthlyCharges'] = df_plot['MonthlyCharges'].astype(int)
    churn_scatter_graph('MonthlyCharges')

    kmeans_churn_graph('MonthlyCharges', 'MonthlyChargesCluster')

    churn_df.loc[pd.to_numeric(churn_df['TotalCharges'], errors='coerce').isnull(),'TotalCharges'] = np.nan
    churn_df = churn_df.dropna()
    churn_df['TotalCharges'] = pd.to_numeric(churn_df['TotalCharges'], errors='coerce')

    df_plot = churn_df.copy()
    df_plot['TotalCharges'] = df_plot['TotalCharges'].astype(int)
    churn_scatter_graph('TotalCharges')
    kmeans_churn_graph('TotalCharges', 'TotalChargeCluster', churn_df.copy())

    le = LabelEncoder()
    dummy_columns = [] #array for multiple value columns
    for column in churn_df.columns:
        if churn_df[column].dtype == object and column != 'customerID':
            if churn_df[column].nunique() == 2:
                #apply Label Encoder for binary ones
                churn_df[column] = le.fit_transform(churn_df[column]) 
            else:
                dummy_columns.append(column)
    #apply get dummies for selected columns
    churn_df = pd.get_dummies(data = churn_df,columns = dummy_columns)

    #st.write(churn_df[['gender','Partner','TenureCluster_High','TenureCluster_Low','TenureCluster_Mid']].head())
    all_columns = []
    for column in churn_df.columns:
        column = column.replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_")
        all_columns.append(column)

    churn_df.columns = all_columns
    glm_columns = 'gender'

    for column in churn_df.columns:
        if column not in ['Churn','customerID','gender']:
            glm_columns = glm_columns + ' + ' + column
    
    glm_model = smf.glm(formula='Churn ~ {}'.format(glm_columns), data=churn_df, family=sm.families.Binomial())
    res = glm_model.fit()
    st.write(res.summary())
    st.write(np.exp(res.params))
    #create feature set and labels
    X = churn_df.drop(['Churn','customerID'],axis=1)
    y = churn_df.Churn
    #train and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=56)
    #building the model & printing the score
    xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, objective= 'binary:logistic',n_jobs=-1).fit(X_train, y_train)
    st.write('Accuracy of XGB classifier on training set: {:.2f}'
        .format(xgb_model.score(X_train, y_train)))
    st.write('Accuracy of XGB classifier on test set: {:.2f}'
        .format(xgb_model.score(X_test[X_train.columns], y_test)))
    y_pred = xgb_model.predict(X_test)
    st.write(classification_report(y_test, y_pred))

    
    fig, ax = plt.subplots(figsize=(10,6))
    plot_importance(xgb_model, ax=ax)
    st.pyplot(plt.show())

   

    churn_df['probability'] = xgb_model.predict_proba(churn_df[X_train.columns])[:,1]
    if st.checkbox('See Probability result of model'):
        st.write(churn_df[['customerID', 'probability']].head(10))
    #Feature Engineering-grouping numerical columns, label encoder, get_dummies


#Predict next day of purchase of customer
def predict_next_purchase_day():

    st.title("Pridicting Next Purchase Day Of Customers")
    st.write("""Predictive analytics helps us a lot on this one. One of the many opportunities
    it can provide is predicting the next purchase day of the customer.
    **What if you know if a customer is likely to make another purchase in 7 days?**""")

    st.markdown("""
    <p>Strategies can be build like:</p>
    <ol>
     <li> No promotional offer to this customers(he/she)</li>
     <li> High LTV</li>
     <li> Mid LTV</li>
    </ol>
    """, unsafe_allow_html= True)
    if st.checkbox("View Data:"):
        data.head(10)

    #create dataframe with uk data only
    uk = data.query("Country=='United Kingdom'").reset_index(drop=True)

    df_np_6m = uk[(uk.InvoiceYearMonth < 201109) &
             (uk.InvoiceYearMonth >= 201103)].reset_index(drop=True)

    df_np_next = uk[(uk.InvoiceYearMonth >= 201109) & 
                (uk.InvoiceYearMonth < 201112)].reset_index(drop=True)

    np_user = pd.DataFrame(df_np_6m['CustomerID'].unique())
    np_user.columns = ['CustomerID']

    #create a dataframe with customer id and first purchase date in tx_next
    next_first_purchase = df_np_next.groupby('CustomerID').InvoiceDate.min().reset_index()
    next_first_purchase.columns = ['CustomerID','MinPurchaseDate']

    #create a dataframe with customer id and last purchase date in tx_6m
    last_purchase = df_np_6m.groupby('CustomerID').InvoiceDate.max().reset_index()
    last_purchase.columns = ['CustomerID','MaxPurchaseDate']

    #merge two dataframes
    purchase_dates = pd.merge(last_purchase,next_first_purchase,on='CustomerID',how='left')

    #calculate the time difference in days:
    purchase_dates['NextPurchaseDay'] = (purchase_dates['MinPurchaseDate'] - purchase_dates['MaxPurchaseDate']).dt.days

    #merge with tx_user 
    np_user = pd.merge(np_user, purchase_dates[['CustomerID','NextPurchaseDay']],on='CustomerID',how='left')

    #print tx_user
    if st.checkbox("View User Data:"):
        st.write(np_user.head())

    #fill NA values with 999
    np_user = np_user.fillna(999)


    #get max purchase date for Recency and create a dataframe
    np_max_purchase = df_np_6m.groupby('CustomerID').InvoiceDate.max().reset_index()
    np_max_purchase.columns = ['CustomerID','MaxPurchaseDate']

    #find the recency in days and add it to tx_user
    np_max_purchase['Recency'] = (np_max_purchase['MaxPurchaseDate'].max() - np_max_purchase['MaxPurchaseDate']).dt.days
    np_user = pd.merge(np_user, np_max_purchase[['CustomerID','Recency']], on='CustomerID')

    #plot recency
    plot_data = [
        go.Histogram(
            x=np_user['Recency']
        )
    ]

    plot_layout = go.Layout(
            xaxis={'title':'reacency'},
            yaxis={'title':'Customer'},
            title='Recency'
        )
    np_fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(np_fig)

    #clustering for Recency
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(np_user[['Recency']])
    np_user['RecencyCluster'] = kmeans.predict(np_user[['Recency']])


    #order recency clusters
    np_user = order_cluster('RecencyCluster', 'Recency',np_user,False)

    #print cluster characteristics
    st.write(np_user.groupby('RecencyCluster')['Recency'].describe())


    #get total purchases for frequency scores
    np_frequency = df_np_6m.groupby('CustomerID').InvoiceDate.count().reset_index()
    np_frequency.columns = ['CustomerID','Frequency']

    #add frequency column to tx_user
    np_user = pd.merge(np_user, np_frequency, on='CustomerID')

    #plot frequency
    plot_data = [
        go.Histogram(
            x=np_user.query('Frequency < 1000')['Frequency']
        )
    ]

    plot_layout = go.Layout(
            title='Frequency'
        )
    st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))
    

    #clustering for frequency
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(np_user[['Frequency']])
    np_user['FrequencyCluster'] = kmeans.predict(np_user[['Frequency']])

    #order frequency clusters and show the characteristics
    np_user = order_cluster('FrequencyCluster', 'Frequency',np_user,True)
    st.write(np_user.groupby('FrequencyCluster')['Frequency'].describe())


    #calculate monetary value, create a dataframe with it
    df_np_6m['Revenue'] = df_np_6m['UnitPrice'] * df_np_6m['Quantity']
    np_revenue = df_np_6m.groupby('CustomerID').Revenue.sum().reset_index()

    #add Revenue column to tx_user
    np_user = pd.merge(np_user, np_revenue, on='CustomerID')

    #plot Revenue
    plot_data = [
        go.Histogram(
            x=np_user.query('Revenue < 10000')['Revenue']
        )
    ]

    plot_layout = go.Layout(
            title='Monetary Value'
        )
    st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))
    

    #Revenue clusters 
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(np_user[['Revenue']])
    np_user['RevenueCluster'] = kmeans.predict(np_user[['Revenue']])

    #ordering clusters and who the characteristics
    np_user = order_cluster('RevenueCluster', 'Revenue',np_user,True)
    np_user.groupby('RevenueCluster')['Revenue'].describe()


    #building overall segmentation
    np_user['OverallScore'] = np_user['RecencyCluster'] + np_user['FrequencyCluster'] + np_user['RevenueCluster']

    #assign segment names
    np_user['Segment'] = 'Low-Value'
    np_user.loc[np_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
    np_user.loc[np_user['OverallScore']>4,'Segment'] = 'High-Value' 

    #plot revenue vs frequency
    np_graph = np_user.query("Revenue < 50000 and Frequency < 2000")

    plot_data = [
        go.Scatter(
            x=np_graph.query("Segment == 'Low-Value'")['Frequency'],
            y=np_graph.query("Segment == 'Low-Value'")['Revenue'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=np_graph.query("Segment == 'Mid-Value'")['Frequency'],
            y=np_graph.query("Segment == 'Mid-Value'")['Revenue'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=np_graph.query("Segment == 'High-Value'")['Frequency'],
            y=np_graph.query("Segment == 'High-Value'")['Revenue'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Revenue"},
            xaxis= {'title': "Frequency"},
            title='Segments'
        )
    st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))
    

    #plot revenue vs recency
    np_graph = np_user.query("Revenue < 50000 and Frequency < 2000")

    plot_data = [
        go.Scatter(
            x=np_graph.query("Segment == 'Low-Value'")['Recency'],
            y=np_graph.query("Segment == 'Low-Value'")['Revenue'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=np_graph.query("Segment == 'Mid-Value'")['Recency'],
            y=np_graph.query("Segment == 'Mid-Value'")['Revenue'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=np_graph.query("Segment == 'High-Value'")['Recency'],
            y=np_graph.query("Segment == 'High-Value'")['Revenue'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Revenue"},
            xaxis= {'title': "Recency"},
            title='Segments'
        )
    st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))
    

    #plot frequency vs recency
    np_graph = np_user.query("Revenue < 50000 and Frequency < 2000")

    plot_data = [
        go.Scatter(
            x=np_graph.query("Segment == 'Low-Value'")['Recency'],
            y=np_graph.query("Segment == 'Low-Value'")['Frequency'],
            mode='markers',
            name='Low',
            marker= dict(size= 7,
                line= dict(width=1),
                color= 'blue',
                opacity= 0.8
            )
        ),
            go.Scatter(
            x=np_graph.query("Segment == 'Mid-Value'")['Recency'],
            y=np_graph.query("Segment == 'Mid-Value'")['Frequency'],
            mode='markers',
            name='Mid',
            marker= dict(size= 9,
                line= dict(width=1),
                color= 'green',
                opacity= 0.5
            )
        ),
            go.Scatter(
            x=np_graph.query("Segment == 'High-Value'")['Recency'],
            y=np_graph.query("Segment == 'High-Value'")['Frequency'],
            mode='markers',
            name='High',
            marker= dict(size= 11,
                line= dict(width=1),
                color= 'red',
                opacity= 0.9
            )
        ),
    ]

    plot_layout = go.Layout(
            yaxis= {'title': "Frequency"},
            xaxis= {'title': "Recency"},
            title='Segments'
        )
    st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))

    #create a dataframe with CustomerID and Invoice Date
    np_day_order = df_np_6m[['CustomerID','InvoiceDate']]

    #convert Invoice Datetime to day
    np_day_order['InvoiceDay'] = df_np_6m['InvoiceDate'].dt.date
    np_day_order = np_day_order.sort_values(['CustomerID','InvoiceDate'])

    #drop duplicates
    np_day_order = np_day_order.drop_duplicates(subset=['CustomerID','InvoiceDay'],keep='first')

    #shifting last 3 purchase dates
    np_day_order['PrevInvoiceDate'] = np_day_order.groupby('CustomerID')['InvoiceDay'].shift(1)
    np_day_order['T2InvoiceDate'] = np_day_order.groupby('CustomerID')['InvoiceDay'].shift(2)
    np_day_order['T3InvoiceDate'] = np_day_order.groupby('CustomerID')['InvoiceDay'].shift(3) 

    st.write(np_day_order.head(5))

    np_day_order['DayDiff'] = (np_day_order['InvoiceDay'] - np_day_order['PrevInvoiceDate']).dt.days
    np_day_order['DayDiff2'] = (np_day_order['InvoiceDay'] - np_day_order['T2InvoiceDate']).dt.days
    np_day_order['DayDiff3'] = (np_day_order['InvoiceDay'] - np_day_order['T3InvoiceDate']).dt.days

    st.write(np_day_order.head())

    np_day_diff = np_day_order.groupby('CustomerID').agg({'DayDiff': ['mean','std']}).reset_index()
    np_day_diff.columns = ['CustomerID', 'DayDiffMean','DayDiffStd']

    np_day_order_last = np_day_order.drop_duplicates(subset=['CustomerID'],keep='last')

    np_day_order_last = np_day_order_last.dropna()
    np_day_order_last = pd.merge(np_day_order_last, np_day_diff, on='CustomerID')
    np_user = pd.merge(np_user, np_day_order_last[['CustomerID','DayDiff','DayDiff2','DayDiff3','DayDiffMean','DayDiffStd']], on='CustomerID')
    
    #create tx_class as a copy of tx_user before applying get_dummies
    np_class = np_user.copy()
    np_class = pd.get_dummies(np_class)
    
    st.write(np_user.NextPurchaseDay.describe())

    np_class['NextPurchaseDayRange'] = 2
    np_class.loc[np_class.NextPurchaseDay>20,'NextPurchaseDayRange'] = 1
    np_class.loc[np_class.NextPurchaseDay>50,'NextPurchaseDayRange'] = 0

    st.write("""
        The **correlation matrix** is one of the cleanest ways to show correlation between our features and label.
    """)

    corr = np_class[np_class.columns].corr()
    plt.figure(figsize = (40,30))
    sns.heatmap(corr, annot = True, linewidths=0.2, fmt=".2f")
    st.pyplot()

    #train & test split
    np_class = np_class.drop('NextPurchaseDay',axis=1)
    X, y = np_class.drop('NextPurchaseDayRange',axis=1), np_class.NextPurchaseDayRange
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=44)

    #create an array of models
    models = []
    models.append(("LR",LogisticRegression(solver='lbfgs')))
    models.append(("NB",GaussianNB()))
    models.append(("RF",RandomForestClassifier()))
    models.append(("SVC",SVC()))
    models.append(("Dtree",DecisionTreeClassifier()))
    models.append(("XGB",xgb.XGBClassifier()))
    models.append(("KNN",KNeighborsClassifier()))

    def save_model(model,path):
        # create a folder named models
        with open("models/"+path+".pk",'wb') as f:
            pickle.dump(model,f)
            print('saved as ',"models/"+path+".pk")


    #measure the accuracy 
    for name,model in models:
        kfold = KFold(n_splits=2, random_state=22)
        cv_result = cross_val_score(model,X_train,y_train, cv = kfold,scoring = "accuracy")
        try:
            model.fit(X_train,y_train)
            save_model(model,name)
        except Exception as e:
            st.write(e)
        st.write(name, cv_result)

    xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
    st.write('Accuracy of XGB classifier on training set: {:.2f}'
        .format(xgb_model.score(X_train, y_train)))
    st.write('Accuracy of XGB classifier on test set: {:.2f}'
       .format(xgb_model.score(X_test[X_train.columns], y_test)))

    
    param_test1 = {
    'max_depth':range(3,10,2),
    'min_child_weight':range(1,6,2)
    }
    gsearch1 = GridSearchCV(estimator = xgb.XGBClassifier(), 
    param_grid = param_test1, scoring='accuracy',n_jobs=-1,iid=False, cv=2)
    gsearch1.fit(X_train,y_train)
    if st.checkbox('Best Grid Parameters'):
        gsearch1.best_params_,
    if st.checkbox('Click to view GridSearch Best Score'):
        'Score: {:.2f}'.format(gsearch1.best_score_* 100)

    if st.checkbox('Enter Input to Predict Customer Next Purchase Day'):
        st.info("Please Fill Data")
        Customer_id = st.number_input('CusomerId',min_value =10000, max_value = 99999, value=12345 )
        Recency_no = st.number_input('RecencyNo', min_value=0, max_value=365, value=123, step=5)
        Recency_cluster = st.number_input('RecencyCluster', min_value=0, max_value=3, value=2)
        Frequency_input = st.number_input('Frequency', min_value=1, max_value=10000, value=5455, step=100)
        Frequency_cluster = st.number_input('FrequencyCluster', min_value=0, max_value=365, value=123, step = 4)
        Revenue = st.number_input('Revenue', min_value=100, max_value=100000, value=9999, step=1000)
        Revenue_cluster = st.number_input('RevenueCluster', min_value=0, max_value=2, value=1)
        OverallScore = st.number_input('OverAllScore', min_value=0, max_value=10, value=6)
        DayDiff = st.number_input('DayDiff', min_value=1, max_value=200, value=121, step = 5)
        DayDiff2 = st.number_input('DayDiff2', min_value=1, max_value=200, value=67, step=5)
        DayDiff3 = st.number_input('DayDiff3', min_value=1, max_value=200, value=72, step=5)
        DayData = [DayDiff + DayDiff2 + DayDiff3]
        DayDiffMean = np.mean(DayData)
        DayDiffStdev = np.std(DayData)
        Segment_high_value = st.number_input('SegmentHighValue', min_value=0, max_value=1, value=1)
        Segment_Low_value = st.number_input('SegmentLowValue', min_value=0, max_value=1, value=0)
        Segment_Mid_value = st.number_input('SegmentMidValue', min_value=0, max_value=1, value=1)
       # Next_Purchase_day_Range = st.number_input('NextPurchaseDayRange', min_value=0, max_value=2, value=1)
        model_name = st.selectbox('Select Model Name:', ["LR", "NB", "RF", "SVC", "Dtree", "XGB" ,"KNN"])

        ls_data = [Customer_id, Recency_no, Recency_cluster, Frequency_input, Frequency_cluster, Revenue, Revenue_cluster, OverallScore, DayDiff, DayDiff2, 
                 DayDiff3, DayDiffMean, DayDiffStdev, Segment_high_value,Segment_Low_value, Segment_Mid_value  ]

        st.write(ls_data,predict_upnext_purchase_day(ls_data, model_name))



#predict sales effect
def predicting_sales():
    st.markdown("""
        <h3>Predicting Future Sales</h3>  
        <p> It can be utilized for planning and also determine 
        demand and supply actions by looking at the forecasts.
         It helps to see where to invest more, 
         it is an excellent guide for planning budgets and targets.</p>

        <p>Time series forecasting is one of the major building blocks of Machine Learning.
        for this purpose we will focus on Long Short-term Memory (LSTM) method.</p> 
    """, 
    unsafe_allow_html=True)
    df_sales = st.cache(pd.read_csv)("Data/Sales_data/sales_train.csv")

    #convert date field from string to datetime
    df_sales['date'] = pd.to_datetime(df_sales['date'])
    if st.checkbox('View data'):
        st.write(df_sales.head())

    #represent month in date field as its first day
    df_sales['date'] = df_sales['date'].dt.year.astype('str') + '-' + df_sales['date'].dt.month.astype('str') + '-01'
    df_sales['date'] = pd.to_datetime(df_sales['date'])
    #groupby date and sum the sales
    df_sales = df_sales.groupby('date').sales.sum().reset_index()
    if st.checkbox('View Data'):
        st.write(df_sales.head())

    st.markdown("""
        <h3>Data Transformation</h3>
        <p>To model our forecast easier and more accurate, 
        we will do the transformations below:</p>
        <ol>
        <li>We will convert the data to stationary if it is not</li>
        <li>Converting from time series to supervised for having the feature set of our LSTM model</li>
        <li>Scale the data</li>
        </ol>
    """,
    unsafe_allow_html=True)

    #plot monthly sales -1
    def predict_sales_scatter(predictdatefield):
        plot_data = [
            go.Scatter(
                x=df_sales[predictdatefield],
                y=df_sales['sales'],
            )
        ]
        plot_layout = go.Layout(
                xaxis={'title':'year'},
                yaxis={'title':'Sales'},
                title='Monthly Sales'
            )
        return st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))

    predict_sales_scatter('date')

    st.write("""It is not stationary and has an increasing trend over the months.""")


    #create a new dataframe to model the difference
    df_diff = df_sales.copy()
    #add previous sales to the next row
    df_diff['prev_sales'] = df_diff['sales'].shift(1)
    #drop the null values and calculate the difference
    df_diff = df_diff.dropna()
    df_diff['diff'] = (df_diff['sales'] - df_diff['prev_sales'])
    df_diff.head(10)

    #plot sales diff 2
    
    predict_sales_scatter('date')

    st.write("""Perfect! Now we can start building our feature set.
    So what we need to do is to create columns from lag_1 to lag_12 and assign values
    by using shift() method:""")

    #create dataframe for transformation from time series to supervised
    df_supervised = df_diff.drop(['prev_sales'],axis=1)
    #adding lags
    for inc in range(1,13):
        field_name = 'lag_' + str(inc)
        df_supervised[field_name] = df_supervised['diff'].shift(inc)
    #drop null values
    df_supervised = df_supervised.dropna().reset_index(drop=True)

    st.write(df_supervised.head()) #feature set

    # Define the regression formula
    model = smf.ols(formula='diff ~ lag_1', data=df_supervised)
    # Fit the regression
    model_fit = model.fit()
    # Extract the adjusted r-squared
    regression_adj_rsq = model_fit.rsquared_adj
    st.write("""Basically, we fit a linear regression model (OLS — Ordinary Least Squares)
    and calculate the Adjusted R-squared. For the example above, we just used lag_1 to see how much 
    it explains the variation in column diff.""")
    st.write("The output:{:,.2f} ".format(regression_adj_rsq * 100))
    st.write("lag_1 explains 3% of the variation. Let's check out for others:")

    model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5', data=df_supervised)
    # Fit the regression
    model_fit = model.fit()
    # Extract the adjusted r-squared
    st.write("Now the output is:{:,.2f} ".format((model_fit.rsquared_adj) * 100))
    st.write("Adding four more features increased the score from 3% to 44%.")

    model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 + lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12 + lag_12', data=df_supervised)
    # Fit the regression
    model_fit = model.fit()
    # Extract the adjusted r-squared
    st.write("Now the output is more accurate: {:,.2f}".format((model_fit.rsquared_adj) * 100))
    st.write("""The result is impressive as the score is increased. Now we can confidently build our model after scaling our data.""")
    df_model = df_supervised.drop(['sales','date'],axis=1)

    #split train and test set -error
    st.write(df_model.head())
    train_set, test_set = df_model[0:-6].values, df_model[-6:].values #selected last 6-months data

    #apply Min Max Scaler
    scaler = MinMaxScaler(feature_range=(-1, 1)) #scale each future between -1 and 1
    scaler = scaler.fit(train_set)
    # reshape training set
    train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
    train_set_scaled = scaler.transform(train_set)
    # reshape test set
    test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
    test_set_scaled = scaler.transform(test_set)

    # create feature and label sets from scaled datasets:
    X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1]
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1]
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    #building the LSTM model

    model = Sequential()
    model.add(LSTM(4, batch_input_shape=(1, X_train.shape[1], X_train.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, nb_epoch=100, batch_size=1, verbose=1, shuffle=False)

    y_pred = model.predict(X_test,batch_size=1)

    st.image(Image.open('y_pred.png'),caption="y_pred", use_column_width=True)
    st.image(Image.open('y_test.png'), caption="y_test", use_column_width=True)

    #Inverse Transformation for scaling
    #reshape y_pred
    y_pred = y_pred.reshape(y_pred.shape[0], 1, y_pred.shape[1])
    #rebuild test set for inverse transform
    pred_test_set = []
    for index in range(0,len(y_pred)):
        st.write(np.concatenate([y_pred[index],X_test[index]],axis=1))
        pred_test_set.append(np.concatenate([y_pred[index],X_test[index]],axis=1))
    #reshape pred_test_set
    pred_test_set = np.array(pred_test_set)
    pred_test_set = pred_test_set.reshape(pred_test_set.shape[0], pred_test_set.shape[2])
    #inverse transform
    pred_test_set_inverted = scaler.inverse_transform(pred_test_set) 
    
    #create dataframe that shows the predicted sales
    result_list = []
    sales_dates = list(df_sales[-7:].date)
    act_sales = list(df_sales[-7:].sales)
    for index in range(0,len(pred_test_set_inverted)):
        result_dict = {}
        result_dict['pred_value'] = int(pred_test_set_inverted[index][0] + act_sales[index])
        result_dict['date'] = sales_dates[index+1]
        result_list.append(result_dict)
    df_result = pd.DataFrame(result_list)

    #merge with actual sales dataframe
    df_sales_pred = pd.merge(df_sales,df_result,on='date',how='left')
    #plot actual and predicted -3
    plot_data = [
        go.Scatter(
            x=df_sales_pred['date'],
            y=df_sales_pred['sales'],
            name='actual'
        ),
            go.Scatter(
            x=df_sales_pred['date'],
            y=df_sales_pred['pred_value'],
            name='predicted'
        )
        
    ]
    plot_layout = go.Layout(
            title='Sales Prediction'
        )

    st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))


#Market Response Model
def market_response_model():
    st.markdown("""

    <h3>Market Response Model</h3>    
    <p>This model will Help in increasing product sales.
    <i>If we do a discount today, how many incremental transactions should we expect?</i>
    Segmenting customers and doing A/B tests enable 
    us to try lots of different ideas for generating incremental sales.
    Let's see an example below:</p>

    """,unsafe_allow_html=True)
    st.image(Image.open("market_response_ex.png"), caption = "Example", use_column_width=True )

    df_response = st.cache(pd.read_csv)("Data/response_data.csv")
    st.write(df_response.head())

    st.markdown("""
        <p>We will be building a binary classification model for scoring the 
        conversion probability of all customers.
        For doing that, we are going to follow the steps below:</p>
        <ol>
        <li>Building the uplift formula</li>
        <li>Exploratory Data Analysis (EDA) & Feature Engineering</li>
        <li>Scoring the conversion probabilities</li>
        <li>Observing the results on the test set</li>
        </ol>
    """, unsafe_allow_html=True)

    st.write("**Uplift Formula**")
    st.latex(r'Conversion Uplift: Conversion Rate of Test Group - Conversion Rate of Control Group')
    st.latex(r'Order Uplift: Conversion Uplift *  Converted Customer in Test Group')
    st.latex(r'Revenue Uplift: Order Uplift * Average Order Value')

    #Uplift Function
    def calc_uplift(df):
        #assigning 25$ to the average order value
        avg_order_value = 25
        
        #calculate conversions for each offer type
        base_conv = df[df.offer == 'No Offer']['conversion'].mean()
        disc_conv = df[df.offer == 'Discount']['conversion'].mean()
        bogo_conv = df[df.offer == 'Buy One Get One']['conversion'].mean()
        
        #calculate conversion uplift for discount and bogo
        disc_conv_uplift = disc_conv - base_conv
        bogo_conv_uplift = bogo_conv - base_conv
        
        #calculate order uplift
        disc_order_uplift = disc_conv_uplift * len(df[df.offer == 'Discount']['conversion'])
        bogo_order_uplift = bogo_conv_uplift * len(df[df.offer == 'Buy One Get One']['conversion'])
        
        #calculate revenue uplift
        disc_rev_uplift = disc_order_uplift * avg_order_value
        bogo_rev_uplift = bogo_order_uplift * avg_order_value
        
        
        st.write('Discount Conversion Uplift: {0}%'.format(np.round(disc_conv_uplift*100,2)))
        st.write('Discount Order Uplift: {0}'.format(np.round(disc_order_uplift,2)))
        st.write('Discount Revenue Uplift: ${0}\n'.format(np.round(disc_rev_uplift,2)))
        st.write('--------------------------------------------------------------')
        if len(df[df.offer == 'Buy One Get One']['conversion']) > 0:
            st.write('BOGO Conversion Uplift: {0}%'.format(np.round(bogo_conv_uplift*100,2)))
            st.write('BOGO Order Uplift: {0}'.format(np.round(bogo_order_uplift,2)))
            st.write('BOGO Revenue Uplift: ${0}'.format(np.round(bogo_rev_uplift,2)))
        st.write("""Discount looks like a better option if we want to get more conversion. 
        It brings {a}% uptick compared to the customers who didn’t receive any offer.
        BOGO (Buy One Get One) has {b}% uptick as well.""".format(a=np.round(disc_conv_uplift*100,2),
                                                    b = np.round(bogo_conv_uplift*100,2)))
        
    st.write(calc_uplift(df_response))

    #1
    st.write("We check every feature one by one to find out their impact on conversion")
    st.write("**1. Recency**")
    def response_bar_graph(response_data_field):
        df_plot = df_response.groupby(response_data_field).conversion.mean().reset_index()
        plot_data = [
            go.Bar(
                x=df_plot[response_data_field],
                y=df_plot['conversion'],
                marker=dict(
                color=['green', 'blue', 'orange', 'yellow', 'red'])
            )
        ]
        plot_layout = go.Layout(
                xaxis={"type": "category"},
                title='{} vs Conversion'.format(response_data_field),
                plot_bgcolor  = 'rgb(243,243,243)',
                paper_bgcolor  = 'rgb(243,243,243)',
            )
        return st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))

    response_bar_graph('recency')
    st.write("It goes as expected until 11 months of recency. Then it increases. ")
    st.write("**2. History**")
    st.write("""We will create a history cluster and observe its impact. Let’s apply 
    k-means clustering to define the significant groups in history:""")

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df_response[['history']])
    df_response['history_cluster'] = kmeans.predict(df_response[['history']])
    #order the cluster numbers 
    df_response = order_cluster('history_cluster', 'history',df_response,True)
    #print how the clusters look like
    df_response.groupby('history_cluster').agg({'history':['mean','min','max'], 'conversion':['count', 'mean']})
    #plot the conversion by each cluster
    df_plot = df_response.groupby('history_cluster').conversion.mean().reset_index()
    plot_data = [
        go.Bar(
            x=df_plot['history_cluster'],
            y=df_plot['conversion'],
        )
    ]
    plot_layout = go.Layout(
            xaxis={"type": "category"},
            title='History vs Conversion',
            plot_bgcolor  = 'rgb(243,243,243)',
            paper_bgcolor  = 'rgb(243,243,243)',
        )

    st.write("Overview of the clusters and the plot vs. conversion:")
    st.write(df_response.groupby('history_cluster').agg({
                                        'history':{'mean', 'min', 'max'},
                                        'conversion':{'count', 'mean'}
    }))
    st.plotly_chart(go.Figure(data=plot_data, layout=plot_layout))
    st.write("Customers with higher $ value of history are more likely to convert.")
    st.write("**3. Used Discount & BOGO**")


    st.write(df_response.groupby(['used_discount','used_bogo','offer']).agg({'conversion':'mean'}))
    st.write("**4. Zip Code**")
    st.write("Rural shows better conversion compared to others:")
    #2
    response_bar_graph('zip_code')
    
    st.write("**5. Referral**")
    st.write("As we see below, customers from referral channel have less conversion rate:")
    
    #3
    response_bar_graph('is_referral') 

    #4
    st.write(' Using more than one channel is an indicator of high engagement.')
    response_bar_graph('channel')

    #5
    response_bar_graph('offer')

    st.write("""Customers who get discount offers show more conversion whereas it is ~15% for BOGO. 
    If customers don’t get an offer, their conversion rate drops by ~4%.""")

    #building machine learning model
    df_model = df_response.copy()
    df_model = pd.get_dummies(df_model)

    #create feature set and labels
    X = df_model.drop(['conversion'],axis=1) # splitting features and the label
    y = df_model.conversion

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)
    xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
    X_test['proba'] = xgb_model.predict_proba(X_test)[:,1]
    X_test['conversion'] = y_test


    st.write(X_test.head())

    real_disc_uptick = len(X_test)*(X_test[X_test['offer_Discount'] == 1].conversion.mean() - X_test[X_test['offer_No Offer'] == 1].conversion.mean())
    pred_disc_uptick = len(X_test)*(X_test[X_test['offer_Discount'] == 1].proba.mean() - X_test[X_test['offer_No Offer'] == 1].proba.mean())

    st.write('Real Discount Uptick - Order: {:,.2f}, Revenue: {:,.2f}'.format(real_disc_uptick, real_disc_uptick*25))
    st.write('Predicted Discount Uptick - Order: {:,.2f}, Revenue: {:,.2f}'.format(pred_disc_uptick, pred_disc_uptick*25))

    real_bogo_uptick = len(X_test)*(X_test[X_test['offer_Buy One Get One'] == 1].conversion.mean() - X_test[X_test['offer_No Offer'] == 1].conversion.mean())
    pred_bogo_uptick = len(X_test)*(X_test[X_test['offer_Buy One Get One'] == 1].proba.mean() - X_test[X_test['offer_No Offer'] == 1].proba.mean())

    st.write('Real Discount Uptick - Order: {:,.2f}, Revenue: {:,.2f}'.format(real_bogo_uptick, real_bogo_uptick*25))
    st.write('Predicted Discount Uptick - Order: {:,.2f}, Revenue: {:,.2f}'.format(pred_bogo_uptick, pred_bogo_uptick*25))

    #Multi-classification Model for Predicting the Uplift Score
    st.markdown("""
    <ul>
    <li>Customers that will purchase only if they receive an offer(TR)</li>
    <li>Customer that won’t purchase in any case(TN)</li>
    <li>Customers that will purchase without an offer(CR)</li>
    <li>Customers that will not purchase if they don’t receive an offer(CN)</li>
    </ul>""", unsafe_allow_html=True)

    df_response['campaign_group'] = 'treatment'
    df_response.loc[df_response.offer == 'No Offer', 'campaign_group'] = 'control'
    df_response['target_class'] = 0 #CN
    df_response.loc[(df_response.campaign_group == 'control') & (df_response.conversion > 0),'target_class'] = 1 #CR
    df_response.loc[(df_response.campaign_group == 'treatment') & (df_response.conversion == 0),'target_class'] = 2 #TN
    df_response.loc[(df_response.campaign_group == 'treatment') & (df_response.conversion > 0),'target_class'] = 3 #TR
    #creating the clusters
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(df_response[['history']])
    df_response['history_cluster'] = kmeans.predict(df_response[['history']])
    #order the clusters
    df_response = order_cluster('history_cluster', 'history',df_response,True)
    #creating a new dataframe as model and dropping columns that defines the label
    df_model = df_response.drop(['offer','campaign_group','conversion'],axis=1)
    #convert categorical columns
    df_model = pd.get_dummies(df_model)


    #create feature set and labels
    X = df_model.drop(['target_class'],axis=1)
    y = df_model.target_class
    #splitting train and test groups
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)
    #fitting the model and predicting the probabilities
    xgb_model = xgb.XGBClassifier().fit(X_train, y_train)
    class_probs = xgb_model.predict_proba(X_test)
    #st.write(class_probs[0] * 100)
    st.write(r'Predicted Probability Score : {:,.2f}'.format(class_probs[0][0] * 100))

    st.latex(r'UpliftScore = TR + CN - TN - CR')

    st.write("Calculating Uplift Score:")
    st.write("Uplift Score = {:,.2f} " .format(class_probs[0][3] * 100 + class_probs[0][0] * 100 - class_probs[0][1] * 100 - class_probs[0][2] * 100 ))

    st.write("Let’s apply this to all users and calculate the uplift score:")
    #probabilities for all customers
    overall_proba = xgb_model.predict_proba(df_model.drop(['target_class'],axis=1))
    #assign probabilities to 4 different columns
    df_model['proba_CN'] = overall_proba[:,0] 
    df_model['proba_CR'] = overall_proba[:,1] 
    df_model['proba_TN'] = overall_proba[:,2] 
    df_model['proba_TR'] = overall_proba[:,3]
    #calculate uplift score for all customers
    df_model['uplift_score'] = df_model.eval('proba_CN + proba_TR - proba_TN - proba_CR')
    #assign it back to main dataframe
    df_response['uplift_score'] = df_model['uplift_score']
    st.write("we added a uplift_score column in our main dataframe and it looks like below:")
    df_response.head()

    response_html = """ 
    <p>Model Evaluation To evaluate our model, we will create two different groups and compare them with our benchmark.
    Groups are:</p>
    <ol>
    <li>1- High Uplift Score: Customers have uplift score > 3rd quantile</li>
    <li>2- Low Uplift Score: Customers have uplift score < 2nd quantile</li>
    <ol>
    <p>We are going to compare Conversion uplift
    Revenue uplift per target customer to see if our model can make our actions more efficient.</p>"""
    st.markdown(response_html,unsafe_allow_html = True)

    df_response_lift = df_response.copy()
    uplift_q_75 = df_response_lift.uplift_score.quantile(0.75)
    df_response_lift = df_response_lift[(df_response_lift.offer != 'Buy One Get One') & (df_response_lift.uplift_score > uplift_q_75)].reset_index(drop=True)
    #calculate the uplift
    st.write("User Count: ", len(df_response_lift))
    calc_uplift(df_response_lift)

    df_response1_lift = df_response.copy()
    uplift_q_5 = df_response1_lift.uplift_score.quantile(0.5)
    df_response1_lift = df_response1_lift[(df_response1_lift.offer != 'Buy One Get One') & (df_response1_lift.uplift_score < uplift_q_5)].reset_index(drop=True)
    #calculate the uplift
    st.write("User Count: ", len(df_response1_lift))
    calc_uplift(df_response1_lift)

    st.write("""By using this model, we can easily make our campaign more efficient by:
    Targeting specific segments based on the uplift score
    Trying different offers based on customer’s uplift score""")


def predict_upnext_purchase_day(data, modelName):
    with open("models/"+modelName+".pk",'rb') as f:
        model = pickle.load(f)
    return model.predict(np.array(data).reshape(1,-1))[0]


if __name__ == "__main__":
    main()