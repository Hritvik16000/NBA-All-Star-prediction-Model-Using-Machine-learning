#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

combined_stats = pd.read_csv("Combined.csv")
pergame = pd.read_csv("Stats Per Game.csv")
awards = pd.read_csv("Award Shares of Players.csv")
allstar = pd.read_csv("All-Star.csv")
team_stats = pd.read_csv("Summaries.csv")


# In[2]:


pergame


# In[3]:


pergame.shape


# In[4]:


pergame.describe()


# In[5]:


import numpy as np

# Assuming 'player' and 'season' columns are present in both 'pergame' and 'allstar' DataFrames

# Creating a set of tuples for faster lookup
star_season_set = set((row['player'], row['season']) for _, row in allstar.iterrows())

# Using np.where for vectorized assignment
pergame['all_star'] = np.where([(row['player'], row['season']) in star_season_set for _, row in pergame.iterrows()], True, False)


# In[6]:


# Calculating win percentage directly using vectorized operations
team_stats['win_perc'] = team_stats['w'] / (team_stats['w'] + team_stats['l'])

# Selecting only the necessary columns
team_stats = team_stats[['season', 'abbreviation', 'win_perc', 'w', 'l']]

# Merging dataframes using pandas merge
team_player_stats = pd.merge(pergame, team_stats, how='left', left_on=['season', 'tm'], right_on=['season', 'abbreviation'])

# Dropping the redundant 'abbreviation' column
team_player_stats.drop('abbreviation', axis=1, inplace=True)


# In[7]:


# Filtering data using boolean indexing
data_modern = team_player_stats[(team_player_stats['season'] >= 2000) & (team_player_stats['season'] <= 2023)]

# Generating descriptive statistics
data_modern_description = data_modern.describe()


# In[8]:


data_modern.head()


# In[9]:


# Converting 'season' column to categorical without warnings
data_modern[data_modern.columns[data_modern.columns.get_loc('season')]] = pd.Categorical(data_modern['season'])

# Dropping rows with missing values in 'win_perc' column
data_modern = data_modern.dropna(subset=['win_perc'])


# In[10]:


east_teams = set(['MIL', 'TOT', 'ATL', 'IND', 'ORL', 'BOS', 'DET', 'CHI', 'BRK', 'WAS',
                  'MIA', 'CHO', 'NYK', 'CLE', 'TOR', 'PHI', 'CHA', 'NJN', 'CHH', 'WSB', 'KCK'])
data_modern['conf'] = data_modern['tm'].apply(lambda x: 'east' if x in east_teams else 'west')


# In[11]:


import plotly.express as px

# Renaming the DataFrame to avoid modifying the original one
data_modern_model = data_modern.copy()

# Creating the scatter plot with custom colors
fig = px.scatter(data_modern_model, x="win_perc", y='pts_per_game', color="all_star",
                 category_orders={"all_star": [True, False]}, labels={"all_star": "All-Star"},
                 size_max=15, hover_name="player", hover_data=["tm"],
                 color_discrete_map={True: "#003f5c", False: "#ffa600"},
                 symbol="all_star", symbol_sequence=["circle", "x"],
                 opacity=0.7)

# Updating layout with larger scatter plot
fig.update_layout(xaxis_title="Win Percentage", yaxis_title="Points Per Game",
                  title='Scoring and Winning relation with All-Star Selection (2000-2023)',
                  yaxis_range=[0, 40], height=600, width=1000,
                  font=dict(family="Arial, sans-serif", size=12),
                  plot_bgcolor="rgba(255, 255, 255, 0.9)",
                  legend=dict(title="", orientation="h", y=1.1, yanchor="bottom"))

# Showing the plot
fig.show()


# In[12]:


import plotly.graph_objects as go

# Filtering data for All-Star players
all_stars = data_modern_model[data_modern_model['all_star'] == True]

# Creating a histogram
fig = px.histogram(all_stars, x='age')

# Adding a line for the average age
fig.add_trace(go.Scatter(x=[all_stars["age"].mean(), all_stars["age"].mean()], y=[0, 140],
                         mode="lines", line=dict(color="#ff7f0e", width=2, dash='dash'),
                         name="Average Age"))

# Updating layout
fig.update_layout(xaxis_title="Age", yaxis_title="Frequency",
                  title=dict(text='Age Distribution of NBA All-Stars (2000-2023)',
                             x=0.5, y=0.95, xanchor='center', yanchor='top'),
                  plot_bgcolor="rgba(255, 255, 255, 0.9)",
                  font=dict(family="Arial, sans-serif", size=12),
                  showlegend=True, legend=dict(title="", orientation="h", y=1.02, yanchor="bottom"),
                  annotations=[dict(x=all_stars["age"].mean(), y=100, xref="x", yref="y",
                                    text="Average Age", showarrow=True, arrowhead=2,
                                    ax=20, ay=-30)])

# Showing the plot
fig.show()


# In[13]:


# Calculate total games played using vectorized addition
data_modern['total_games'] = data_modern['w'] + data_modern['l']

# Calculate percentage of games played using vectorized division
data_modern['games_pct'] = data_modern['g'] / data_modern['total_games']



# In[14]:


# Make a copy of the original DataFrame
data_model1 = data_modern.copy()

# Fill missing values with column medians for numeric columns
data_model1 = data_model1.fillna(data_model1.median(numeric_only=True))



# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

def transform(x_cols):
    return make_column_transformer(
                (OneHotEncoder(), ['pos']),
                (StandardScaler(), x_cols[1:]), # Everything except for 'pos'
    )


# In[16]:


import random

def get_train_test(data, test_size):
    global train
    global test
    train_years = list(range(2000,2023))
    test_years = []
    for i in range(test_size):
        year = random.choice(train_years)
        train_years.remove(year)
        test_years.append(year)

    train = data.loc[data['season'].isin(train_years)]
    test = data.loc[data['season'].isin(test_years)]

    print("Test Seasons:")
    print(test_years)

get_train_test(data_model1, 6)


# In[17]:


# x_cols = ["pos", "experience", "games_pct", "e_fg_percent", "ft_percent", "orb_per_game","drb_per_game", "trb_per_game",
#          "ast_per_game", "stl_per_game", "blk_per_game", "tov_per_game", "pf_per_game","pts_per_game", "win_perc",
#          "ft_per_game", "ts_percent", "per", "usg_percent", "ws", "vorp", "all_star_appearances"]

x_cols = ["pos", "experience", "games_pct", "e_fg_percent", "ft_percent", "orb_per_game","drb_per_game", "trb_per_game",
         "ast_per_game", "stl_per_game", "blk_per_game", "tov_per_game", "pf_per_game","pts_per_game", "win_perc",
         "ft_per_game"]
trans = transform(x_cols)

y_col = "all_star"

model = Pipeline([
            ("tr", trans),
            ("lr", LogisticRegression(fit_intercept = False, max_iter = 1000)),
        ])

model.fit(train[x_cols], train[y_col])
model.score(test[x_cols], test[y_col])


# In[18]:


import numpy as np

def rank_predict(test, model):
    test["prob_all_star"] = model.predict_proba(test[x_cols])[:, 1]
    test['rank'] = test.groupby(['season','conf'])['prob_all_star'].rank(ascending=False)
    test['pred_all_star'] = np.where(  (test['rank'] <= 12.0 ), True, False)

rank_predict(test, model)


# In[19]:


def score_model(test):
    predictions = list(test['all_star'])
    actual = list(test['all_star'])
    total = len(test)
    correct = 0
    for i in range(len(test)):
        if predictions[i] == actual[i]:
            correct += 1
    return correct / total



# In[20]:


# Compute the total number of players classified as All-Stars in the data
total_all_stars_actual = test['all_star'].sum()

# Compute the total number of players predicted as All-Stars
total_all_stars_predicted = test['pred_all_star'].sum()

# Print the results
print("Total players classified as All-Stars in data:", total_all_stars_actual)
print("Total players predicted as All-Stars (24 per season):", total_all_stars_predicted)



# In[21]:


from tabulate import tabulate

x = ["pos:center", "pos:pf", "pos:pg", "pos:sf", "pos:sg", "experience", "g", "e_fg_percent", "ft_percent", "orb_per_game",
     "drb_per_game", "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game", "tov_per_game", "pf_per_game",
     "pts_per_game", "win_perc", "ft_per_game"]

coef = model.named_steps['lr'].coef_[0]

# Create a list of lists for table rows
table_data = [[feature, coefficient] for feature, coefficient in zip(x, coef)]

# Print the table using tabulate
print(tabulate(table_data, headers=["Feature", "Coefficient"], tablefmt="grid"))


# In[22]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate confusion matrix and normalize
cm = confusion_matrix(test['all_star'], test['pred_all_star'])
cm_norm = cm / cm.sum(axis=1)[:, np.newaxis]

# Create a colorful heatmap with seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, cmap="viridis", fmt=".2f", linewidths=0.5, linecolor='lightgray', 
            xticklabels=model['lr'].classes_, yticklabels=model['lr'].classes_, cbar=False)

# Set title and labels
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted All-Star', fontsize=12)
plt.ylabel('True All-Star', fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()


# In[23]:


import plotly.graph_objs as go
import numpy as np
from sklearn.metrics import confusion_matrix

# Assuming cm_norm is your normalized confusion matrix

# Get class labels
classes = model['lr'].classes_

# Create a 3D surface plot
data = [
    go.Surface(
        z=cm_norm,
        x=classes,
        y=classes,
        colorscale='Viridis'
    )
]

# Set layout
layout = go.Layout(
    title='3D Confusion Matrix',
    scene=dict(
        xaxis=dict(title='Predicted All-Star'),
        yaxis=dict(title='True All-Star'),
        zaxis=dict(title='Normalized Count')
    ),
    width=800,  # Set width of the plot
    height=600,  # Set height of the plot
    margin=dict(l=0, r=0, b=0, t=40)  # Set margins
)

# Create figure
fig = go.Figure(data=data, layout=layout)

# Show plot
fig.show()


# In[24]:


from sklearn.metrics import log_loss

# Get the predicted probabilities
predicted_probabilities = model.predict_proba(test[x_cols])[:, 1]

# Calculate the log loss
log_loss_score = log_loss(test['all_star'], predicted_probabilities)
print("Log Loss:", log_loss_score)


# In[25]:


data_modern['all_star_appearances'] = data_modern.groupby('player')['all_star'].cumsum() - data_modern['all_star']


# In[26]:


stats = combined_stats[['ts_percent','per','usg_percent','ws','vorp','player','season']]
data_modern = pd.merge(data_modern, stats,  how='left', left_on=['player','season'], right_on = ['player','season'])
data_model2 = data_modern.copy()
data_model2 = data_model2[data_model2['season'] != 2023]
data_model2 = data_model2.fillna(data_model2.median(numeric_only=True))
data_model2
# stats = combined_stats[['ts_percent','per','usg_percent','ws','vorp','player','season']] #apan ne iss file ko rename karke combined stats kara hai advanced stats se
# data_model2 = data_modern.join(stats.set_index(['player','season']), on=['player','season'], how='left')
# data_model2 = data_model2.fillna(data_model2.median(numeric_only=True))
# data_model2


# In[27]:


x_cols = ["pos", "experience", "games_pct", "e_fg_percent", "ft_percent", "orb_per_game","drb_per_game", "trb_per_game",
         "ast_per_game", "stl_per_game", "blk_per_game", "tov_per_game", "pf_per_game","pts_per_game", "win_perc",
         "ft_per_game", "ts_percent", "per", "usg_percent", "ws", "vorp", "all_star_appearances"]
trans = transform(x_cols)

model2 = Pipeline([
            ("tr", trans),
            ("lr", LogisticRegression(fit_intercept = False, max_iter = 1000)),
        ])

logloss2 = []
scores2 = []
for i in range(50):
    get_train_test(data_model2, 6)
    model2.fit(train[x_cols], train[y_col])
    rank_predict(test, model2)
    scores2.append(score_model(test))
    logloss2.append(log_loss(test['all_star'], test["prob_all_star"]))


# In[28]:


x_cols = ["pos", "experience", "games_pct", "e_fg_percent", "ft_percent", "orb_per_game","drb_per_game", "trb_per_game",
         "ast_per_game", "stl_per_game", "blk_per_game", "tov_per_game", "pf_per_game","pts_per_game", "win_perc",
         "ft_per_game"]
trans = transform(x_cols)

logloss1 = []
scores1 = []
for i in range(50):
    get_train_test(data_model1, 6)
    model.fit(train[x_cols], train[y_col])
    rank_predict(test, model)
    scores1.append(score_model(test))  # Modified to pass true and predicted labels
    logloss1.append(log_loss(test['all_star'], test["prob_all_star"]))
    print("Score: " + str(score_model(test)))  # Modified to pass true and predicted labels


# In[29]:


from statistics import mean

print("Average score for original inputs: " + str(mean(scores1)))
print("Average score for new inputs: " + str(mean(scores2)))
print("Average log loss for original inputs: " + str(mean(logloss1)))
print("Average log loss for new inputs: " + str(mean(logloss2)))


# In[30]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss

# Define column transformer
def transform(x_cols):
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['pos']),
            ('num', StandardScaler(), x_cols[1:])  # Everything except for 'pos'
        ])

# Define function to split data into train and test sets
def get_train_test(data, test_size):
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    return train, test

# Define function to train and evaluate model
def train_evaluate_model(train, test, x_cols, y_col):
    trans = transform(x_cols)
    model = Pipeline([
        ("tr", trans),
        ("lr", LogisticRegression(fit_intercept=False, max_iter=1000)),
    ])
    model.fit(train[x_cols], train[y_col])
    predictions = model.predict(test[x_cols])
    probabilities = model.predict_proba(test[x_cols])[:, 1]
    accuracy = accuracy_score(test[y_col], predictions)
    logloss = log_loss(test[y_col], probabilities)
    return accuracy, logloss

# Define columns and train-test split
x_cols = ["pos", "experience", "games_pct", "e_fg_percent", "ft_percent", "orb_per_game",
          "drb_per_game", "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game",
          "tov_per_game", "pf_per_game", "pts_per_game", "win_perc", "ft_per_game"]
y_col = "all_star"
train, test = get_train_test(data_model1, 0.2)

# Train and evaluate model
accuracy, logloss = train_evaluate_model(train, test, x_cols, y_col)
print("Accuracy:", accuracy)
print("Log Loss:", logloss)


# In[31]:


from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Define column transformer
def transform(x_cols):
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['pos']),
            ('num', StandardScaler(), x_cols[1:])  # Everything except for 'pos'
        ])

# Define function to split data into train and test sets
def get_train_test(data, test_size):
    train, test = train_test_split(data, test_size=test_size, random_state=42)
    return train, test

# Define function to train and evaluate model
def train_evaluate_model(train, test, x_cols, y_col):
    trans = transform(x_cols)
    model = Pipeline([
        ("tr", trans),
        ("lr", LogisticRegression(fit_intercept=False, max_iter=1000)),
    ])
    model.fit(train[x_cols], train[y_col])
    return model

# Define columns and train-test split
x_cols = ["pos", "experience", "games_pct", "e_fg_percent", "ft_percent", "orb_per_game",
          "drb_per_game", "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game",
          "tov_per_game", "pf_per_game", "pts_per_game", "win_perc", "ft_per_game",
          "ts_percent", "per", "usg_percent", "ws", "vorp", "all_star_appearances"]
y_col = "all_star"
train, test = get_train_test(data_model2, 0.2)

# Train and evaluate model
model2 = train_evaluate_model(train, test, x_cols, y_col)

# Get coefficients
coef = list(model2.named_steps['lr'].coef_[0])

# Feature names
x = ["pos:center", "pos:pf", "pos:pg", "pos:sf", "pos:sg", "experience", "g", "e_fg_percent",
     "ft_percent", "orb_per_game", "drb_per_game", "trb_per_game", "ast_per_game",
     "stl_per_game", "blk_per_game", "tov_per_game", "pf_per_game", "pts_per_game",
     "win_perc", "ft_per_game", "ts_percent", "per", "usg_percent", "ws", "vorp",
     "all_star_appearances"]

# Combine feature names and coefficients
table_data = list(zip(x, coef))

# Print the table using tabulate
print(tabulate(table_data, headers=['Feature', 'Coefficient'], tablefmt='pretty'))


# In[32]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Define column transformer
def transform(x_cols):
    return ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['pos']),
            ('num', StandardScaler(), x_cols[1:])  # Everything except for 'pos'
        ])

# Define function to train model with GridSearchCV
def train_model_with_gridsearch(data, x_cols, y_col):
    trans = transform(x_cols)
    model = Pipeline([
        ("tr", trans),
        ("lr", LogisticRegression(fit_intercept=False, max_iter=1000)),
    ])
    param_grid = [
        {"lr__C": np.logspace(-5, 5, 20)},
    ]
    clf = GridSearchCV(model, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)
    clf.fit(data[x_cols], data[y_col])
    return clf.best_estimator_

# Define columns
x_cols = ["pos", "experience", "games_pct", "e_fg_percent", "ft_percent", "orb_per_game",
          "drb_per_game", "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game",
          "tov_per_game", "pf_per_game", "pts_per_game", "win_perc", "ft_per_game",
          "ts_percent", "per", "usg_percent", "ws", "vorp", "all_star_appearances"]
y_col = "all_star"

# Train model with GridSearchCV
best_model = train_model_with_gridsearch(data_model2, x_cols, y_col)
best_model


# In[33]:


from sklearn.metrics import log_loss
import numpy as np

# Define the logistic regression model with specified C
model3 = Pipeline([
    ("tr", trans),
    ("lr", LogisticRegression(fit_intercept=False, max_iter=1000, C=0.5455)),
])

# Initialize lists to store scores and log loss
scores3 = []
logloss3 = []

# Loop for 100 iterations
for i in range(100):
    # Get train and test data
    get_train_test(data_model2, 6)

    # Fit the model
    model3.fit(train[x_cols], train[y_col])

    # Make predictions and compute metrics
    rank_predict(test, model3)
    scores3.append(score_model(test))
    logloss3.append(log_loss(test['all_star'], test["prob_all_star"]))


# In[34]:


print("MODEL 2:")
print("AVG Score: " + str(np.mean(scores2)))
print("AVG Log Loss: " + str(np.mean(logloss2)))
print("MODEL 3:")
print("AVG Score: " + str(np.mean(scores3)))
print("AVG Log Loss: " + str(np.mean(logloss3)))


# In[35]:


# Define columns
x_cols = ["pos", "experience", "games_pct", "e_fg_percent", "ft_percent", "orb_per_game",
          "drb_per_game", "trb_per_game", "ast_per_game", "stl_per_game", "blk_per_game",
          "tov_per_game", "pf_per_game", "pts_per_game", "win_perc", "ft_per_game",
          "ts_percent", "per", "usg_percent", "ws", "vorp", "all_star_appearances"]
y_col = "all_star"

finalmodel = Pipeline([
            ("tr", trans),
            ("lr", LogisticRegression(fit_intercept = False, max_iter = 1000, C = 0.5455594781168515, class_weight = "balanced")),
        ])
data_modern = data_modern.fillna(data_modern.median(numeric_only=True))
train = data_modern[data_modern['season'] != 2023]
test = data_modern[data_modern['season'] == 2023]
finalmodel.fit(train[x_cols], train[y_col])
rank_predict(test, finalmodel)


# In[36]:


print(train.columns)


# In[37]:


print(data_modern.columns)


# In[38]:


print(test.shape)


# In[39]:


# Convert 'rank' column to integer type
test['rank'] = test['rank'].astype(int)

# Print 2023 All-Star predictions
print("ALL-STAR PREDICTIONS:")

# Print predictions for the Eastern Conference
print("Eastern Conference:")
east_predictions = test[(test["pred_all_star"] == True) & (test["conf"] == 'east')].sort_values('rank')[['player','rank']]
print(east_predictions.to_string(index=False))

# Print predictions for the Western Conference
print("\nWestern Conference:")
west_predictions = test[(test["pred_all_star"] == True) & (test["conf"] == 'west')].sort_values('rank')[['player','rank']]
print(west_predictions.to_string(index=False))


# In[40]:


print(test[(test['rank'] < 16) & (test['rank'] > 12)].sort_values('rank')[['player', 'conf', 'rank']].to_string(index=False))


# In[41]:


from sklearn.metrics import accuracy_score

# Assuming you have already trained your final model named 'finalmodel' and made predictions on the test data
# Assuming you have true labels for the test data

# True labels for the test data
true_labels = test['all_star']

# Predicted labels by the final model
predicted_labels = test['pred_all_star']

# Calculate accuracy
total_accuracy = accuracy_score(true_labels, predicted_labels)

print("Total Model Accuracy:", total_accuracy)

